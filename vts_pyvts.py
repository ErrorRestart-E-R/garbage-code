"""
pyvts 기반 VTube Studio Public API 어댑터.

프로젝트 내부에서는 `VTubeStudioClient`(직접 WS) 또는 이 어댑터 중 하나를 사용합니다.
`services/lipsync.py`는 아래 메서드/프로퍼티만 기대합니다:
- connected: bool
- ensure_authenticated() -> bool
- inject_parameter(...)
- list_hotkeys_in_current_model(...)
- trigger_hotkey(...)
- close()
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse


@dataclass
class PyVTSAdapterConfig:
    """
    pyvts는 기본적으로 token 파일 경로를 사용해 토큰을 저장/로드합니다.
    (env로 토큰을 넘길 수도 있지만, 파일 저장을 병행하면 운영이 편합니다.)
    """

    ws_url: str
    plugin_name: str
    plugin_developer: str
    token_path: str = "./vts_token.txt"


def _parse_ws_url(ws_url: str) -> Tuple[str, int]:
    """
    ws://localhost:8001 형태를 pyvts(host, port)로 변환.
    """
    u = urlparse(ws_url)
    host = u.hostname or "localhost"
    port = int(u.port or 8001)
    return host, port


class PyVTSClientAdapter:
    def __init__(self, plugin_name: str, plugin_developer: str, ws_url: str, token_path: str = "./vts_token.txt"):
        self.cfg = PyVTSAdapterConfig(
            ws_url=ws_url,
            plugin_name=plugin_name,
            plugin_developer=plugin_developer,
            token_path=token_path or "./vts_token.txt",
        )

        self._vts: Optional[Any] = None
        self._auth_lock = asyncio.Lock()
        self._authenticated = False

    @property
    def connected(self) -> bool:
        return bool(self._vts) and bool(getattr(self._vts, "get_connection_status")())

    async def connect(self) -> None:
        if self.connected:
            return

        # Lazy import: optional dependency
        import pyvts  # type: ignore

        host, port = _parse_ws_url(self.cfg.ws_url)
        plugin_info = {
            "plugin_name": self.cfg.plugin_name,
            "developer": self.cfg.plugin_developer,
            "authentication_token_path": self.cfg.token_path,
        }

        self._vts = pyvts.vts(plugin_info=plugin_info, host=host, port=port)
        await self._vts.connect()

    async def close(self) -> None:
        if self._vts:
            try:
                await self._vts.close()
            finally:
                self._vts = None

    async def ensure_authenticated(self, force_token_request: bool = False) -> bool:
        """
        - 토큰이 없으면: AuthenticationTokenRequest로 발급(사용자 Allow 필요) 후 파일 저장
        - 토큰이 있으면: AuthenticationRequest로 세션 인증
        - 락을 사용해 동시 인증 시도를 방지합니다.
        """
        # 이미 인증됨 (빠른 경로)
        if self._authenticated and self._vts and self._vts.get_authentic_status() == 2:
            return True

        async with self._auth_lock:
            # 락 획득 후 다시 확인 (다른 코루틴이 인증 완료했을 수 있음)
            if self._authenticated and self._vts and self._vts.get_authentic_status() == 2:
                return True

            try:
                if not self._vts or not self.connected:
                    await self.connect()
                assert self._vts is not None

                # 1) 토큰 확보 (토큰 파일 로드 시도)
                if not getattr(self._vts, "authentic_token", None):
                    await self._vts.read_token()

                if not getattr(self._vts, "authentic_token", None) or force_token_request:
                    await self._vts.request_authenticate_token(force=force_token_request)

                # 2) 세션 인증
                resp = await self._vts.request_authenticate()
                # pyvts returns dict or bool depending on version
                if isinstance(resp, dict):
                    ok = resp.get("data", {}).get("authenticated", False)
                else:
                    ok = bool(resp)

                self._authenticated = ok
                return ok
            except Exception as e:
                print(f"VTS auth error: {e}")
                self._authenticated = False
                return False

    async def inject_parameter(self, param_id: str, value: float, mode: str = "set", face_found: bool = False) -> None:
        # 인증되지 않으면 무시 (lipsync는 고빈도로 호출되므로 여기서 인증 시도하지 않음)
        if not self._authenticated:
            return
        if not self._vts or not self.connected:
            return

        try:
            msg = self._vts.vts_request.requestSetParameterValue(
                parameter=param_id,
                value=float(value),
                face_found=bool(face_found),
                mode=str(mode),
            )
            await self._vts.request(msg)
        except Exception:
            # 연결 끊김 등 - 무시하고 다음 프레임에서 재시도
            pass

    async def list_hotkeys_in_current_model(self, model_id: str = "", live2d_item_file_name: str = "") -> Dict[str, Any]:
        """
        pyvts는 기본적으로 current model의 hotkey 목록을 반환합니다.
        (model_id / item_file_name은 현재 pyvts 구현에서 사용하지 않으므로 확장 포인트로만 유지)
        """
        if not self._vts or not self.connected:
            await self.connect()
        assert self._vts is not None

        if self._vts.get_authentic_status() != 2:
            ok = await self.ensure_authenticated()
            if not ok:
                return {}

        msg = self._vts.vts_request.requestHotKeyList()
        return await self._vts.request(msg)

    async def trigger_hotkey(self, hotkey_id_or_name: str, item_instance_id: str = "") -> Dict[str, Any]:
        if not self._vts or not self.connected:
            await self.connect()
        assert self._vts is not None

        if self._vts.get_authentic_status() != 2:
            ok = await self.ensure_authenticated()
            if not ok:
                return {}

        msg = self._vts.vts_request.requestTriggerHotKey(hotkeyID=str(hotkey_id_or_name), itemInstanceID=item_instance_id or None)
        return await self._vts.request(msg)


