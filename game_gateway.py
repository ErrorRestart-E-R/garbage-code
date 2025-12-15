"""
GameHub HTTP client (main core side).

Main core responsibilities:
- Keep persona/voice pipeline consistent
- Ask GameHub for a prompt patch (system_addendum/context_blocks)
- Maintain game session scope per Discord voice channel
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
import asyncio

class GameHubError(RuntimeError):
    pass


@dataclass(frozen=True)
class GameInfo:
    id: str
    name: str
    aliases: List[str]
    description: str


@dataclass(frozen=True)
class ContextBlock:
    title: str
    content: str


@dataclass(frozen=True)
class PrepareTurnPatch:
    active_game_id: Optional[str]
    system_addendum: str
    context_blocks: List[ContextBlock]
    allowed_controls: List[str]
    trace_id: str


class GameHubClient:
    def __init__(
        self,
        base_url: str,
        timeout_total_seconds: float = 2.5,
        timeout_connect_seconds: float = 0.6,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.timeout_total_seconds = float(timeout_total_seconds)
        self.timeout_connect_seconds = float(timeout_connect_seconds)

    def _timeout(self) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(
            total=self.timeout_total_seconds,
            connect=self.timeout_connect_seconds,
        )

    async def _request_json(self, method: str, path: str, json_body: Optional[dict] = None) -> Dict[str, Any]:
        if not self.base_url:
            raise GameHubError("GameHub base_url is empty")

        url = f"{self.base_url}{path}"
        try:
            async with aiohttp.ClientSession(timeout=self._timeout()) as session:
                async with session.request(method, url, json=json_body) as resp:
                    # raise for non-2xx
                    if resp.status < 200 or resp.status >= 300:
                        try:
                            detail = await resp.text()
                        except Exception:
                            detail = ""
                        raise GameHubError(f"GameHub HTTP {resp.status} {method} {path}: {detail}")
                    return await resp.json()
        except asyncio.TimeoutError as e:  # type: ignore[name-defined]
            raise GameHubError(f"GameHub timeout: {method} {path}") from e
        except aiohttp.ClientError as e:
            raise GameHubError(f"GameHub connection error: {method} {path}") from e

    async def health(self) -> bool:
        try:
            await self._request_json("GET", "/health")
            return True
        except Exception:
            return False

    async def list_games(self) -> List[GameInfo]:
        data = await self._request_json("GET", "/v1/games")
        games = []
        for g in (data.get("games") or []):
            try:
                games.append(
                    GameInfo(
                        id=str(g.get("id") or "").strip().lower(),
                        name=str(g.get("name") or "").strip(),
                        aliases=list(g.get("aliases") or []),
                        description=str(g.get("description") or "").strip(),
                    )
                )
            except Exception:
                continue
        return games

    async def start_game(self, session_id: str, game_id: str) -> Optional[str]:
        data = await self._request_json(
            "POST",
            f"/v1/sessions/{session_id}/start",
            json_body={"game_id": game_id},
        )
        gid = (data.get("active_game_id") or "").strip().lower()
        return gid or None

    async def stop_game(self, session_id: str) -> Optional[str]:
        data = await self._request_json("POST", f"/v1/sessions/{session_id}/stop", json_body={})
        gid = (data.get("active_game_id") or "").strip().lower()
        return gid or None

    async def prepare_turn(
        self,
        session_id: str,
        last_user_text: str,
        recent_turns: List[str],
        active_game_id: Optional[str],
    ) -> PrepareTurnPatch:
        data = await self._request_json(
            "POST",
            f"/v1/sessions/{session_id}/prepare_turn",
            json_body={
                "last_user_text": last_user_text,
                "recent_turns": recent_turns or [],
                "active_game_id": active_game_id,
            },
        )
        blocks: List[ContextBlock] = []
        for b in (data.get("context_blocks") or []):
            try:
                blocks.append(
                    ContextBlock(
                        title=str(b.get("title") or ""),
                        content=str(b.get("content") or ""),
                    )
                )
            except Exception:
                continue

        return PrepareTurnPatch(
            active_game_id=(data.get("active_game_id") or None),
            system_addendum=str(data.get("system_addendum") or ""),
            context_blocks=blocks,
            allowed_controls=list(data.get("allowed_controls") or []),
            trace_id=str(data.get("trace_id") or ""),
        )


