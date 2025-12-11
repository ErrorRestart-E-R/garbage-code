import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp


@dataclass
class VTSPluginInfo:
    plugin_name: str
    plugin_developer: str
    auth_token: str = ""


class VTubeStudioClient:
    """
    Minimal VTube Studio Public API client (WebSocket).

    API basics (from DenchiSoft/VTubeStudio README):
    - WebSocket: ws://localhost:8001
    - apiName: "VTubeStudioPublicAPI"
    - apiVersion: "1.0"
    - AuthenticationTokenRequest (1회, 사용자가 VTS에서 Allow)
    - AuthenticationRequest (세션마다, 토큰 사용)
    - InjectParameterDataRequest (파라미터 주입; 최소 1초에 1번 이상 계속 보내야 제어 유지)
    """

    def __init__(self, ws_url: str, plugin: VTSPluginInfo):
        self.ws_url = ws_url
        self.plugin = plugin

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._pending: Dict[str, asyncio.Future] = {}

        self._connected = False
        self._authenticated = False

    @property
    def connected(self) -> bool:
        return self._connected and self._ws is not None and not self._ws.closed

    @property
    def authenticated(self) -> bool:
        return self._authenticated

    async def connect(self) -> None:
        if self.connected:
            return

        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(self.ws_url, heartbeat=20)
        self._connected = True
        self._authenticated = False

        self._listener_task = asyncio.create_task(self._listener_loop())

    async def close(self) -> None:
        self._authenticated = False
        self._connected = False

        if self._listener_task:
            self._listener_task.cancel()
            self._listener_task = None

        # fail pending
        for rid, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(RuntimeError("VTS client closed"))
            self._pending.pop(rid, None)

        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None

        if self._session:
            await self._session.close()
        self._session = None

    async def _listener_loop(self) -> None:
        try:
            assert self._ws is not None
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue

                    rid = data.get("requestID")
                    if rid and rid in self._pending:
                        fut = self._pending.pop(rid)
                        if not fut.done():
                            fut.set_result(data)
                    # else: event/unsolicited response -> ignore
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                    break
        finally:
            self._connected = False
            self._authenticated = False

    async def request(self, message_type: str, data: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
        if not self.connected:
            await self.connect()
        assert self._ws is not None

        request_id = str(uuid.uuid4())
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": request_id,
            "messageType": message_type,
            "data": data,
        }

        fut = asyncio.get_running_loop().create_future()
        self._pending[request_id] = fut
        await self._ws.send_str(json.dumps(payload))

        return await asyncio.wait_for(fut, timeout=timeout)

    async def send(self, message_type: str, data: Dict[str, Any]) -> None:
        """
        Fire-and-forget send. Response will still arrive and be drained by listener loop.
        """
        if not self.connected:
            await self.connect()
        assert self._ws is not None

        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": str(uuid.uuid4()),
            "messageType": message_type,
            "data": data,
        }
        await self._ws.send_str(json.dumps(payload))

    async def ensure_authenticated(self) -> bool:
        """
        Authenticate for current session (requires auth_token).
        If token missing, request token (requires user approval in VTS UI).
        Returns True if authenticated.
        """
        if self._authenticated and self.connected:
            return True

        if not self.connected:
            await self.connect()

        if not self.plugin.auth_token:
            # Request token (user must click Allow in VTS)
            resp = await self.request(
                "AuthenticationTokenRequest",
                {
                    "pluginName": self.plugin.plugin_name,
                    "pluginDeveloper": self.plugin.plugin_developer,
                    # pluginIcon optional
                },
                timeout=60.0,
            )
            token = (resp.get("data") or {}).get("authenticationToken", "")
            if token:
                self.plugin.auth_token = token
            else:
                self._authenticated = False
                return False

        auth_resp = await self.request(
            "AuthenticationRequest",
            {
                "pluginName": self.plugin.plugin_name,
                "pluginDeveloper": self.plugin.plugin_developer,
                "authenticationToken": self.plugin.auth_token,
            },
            timeout=10.0,
        )
        authenticated = bool((auth_resp.get("data") or {}).get("authenticated", False))
        self._authenticated = authenticated
        return authenticated

    async def inject_parameter(self, param_id: str, value: float, mode: str = "set", face_found: bool = False) -> None:
        """
        InjectParameterDataRequest
        Payload example (from DenchiSoft/VTubeStudio README):
        {
          "data": {
            "faceFound": false,
            "mode": "set",
            "parameterValues": [ { "id": "FaceAngleX", "value": 12.31 } ]
          }
        }
        """
        await self.send(
            "InjectParameterDataRequest",
            {
                "faceFound": face_found,
                "mode": mode,
                "parameterValues": [{"id": param_id, "value": float(value)}],
            },
        )

    async def list_hotkeys_in_current_model(self, model_id: str = "", live2d_item_file_name: str = "") -> Dict[str, Any]:
        """
        HotkeysInCurrentModelRequest
        (From DenchiSoft/VTubeStudio README)

        Request payload:
        { "messageType": "HotkeysInCurrentModelRequest",
          "data": { "modelID": "Optional_UniqueIDOfModel",
                    "live2DItemFileName": "Optional_Live2DItemFileName" } }

        Notes:
        - If data omitted, returns hotkeys for current model.
        - If modelID provided, returns hotkeys for that model (if available).
        - live2DItemFileName is for item hotkeys; for now we primarily use current model.
        """
        data: Dict[str, Any] = {}
        if model_id:
            data["modelID"] = model_id
        if live2d_item_file_name:
            data["live2DItemFileName"] = live2d_item_file_name

        return await self.request("HotkeysInCurrentModelRequest", data, timeout=10.0)

    async def trigger_hotkey(self, hotkey_id_or_name: str, item_instance_id: str = "") -> Dict[str, Any]:
        """
        HotkeyTriggerRequest
        (From DenchiSoft/VTubeStudio README)

        Request payload:
        { "messageType": "HotkeyTriggerRequest",
          "data": { "hotkeyID": "HotkeyNameOrUniqueIdOfHotkeyToExecute",
                    "itemInstanceID": "Optional_ItemInstanceIdOfLive2DItemToTriggerThisHotkeyFor" } }
        """
        data: Dict[str, Any] = {"hotkeyID": hotkey_id_or_name}
        if item_instance_id:
            data["itemInstanceID"] = item_instance_id

        return await self.request("HotkeyTriggerRequest", data, timeout=10.0)


