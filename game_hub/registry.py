"""
Game plugin registry for GameHub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from .protocol import ContextBlock, GameInfo


@dataclass(frozen=True)
class PrepareTurnResult:
    system_addendum: str
    context_blocks: List[ContextBlock]
    allowed_controls: List[str]


class GamePlugin(Protocol):
    """
    Minimal plugin contract.
    """

    def info(self) -> GameInfo:
        ...

    def prepare_turn(
        self,
        session_id: str,
        last_user_text: str,
        recent_turns: List[str],
    ) -> PrepareTurnResult:
        ...


class GameRegistry:
    def __init__(self) -> None:
        self._plugins: Dict[str, GamePlugin] = {}

    def register(self, plugin: GamePlugin) -> None:
        gi = plugin.info()
        gid = (gi.id or "").strip().lower()
        if not gid:
            raise ValueError("Game plugin id is empty")
        self._plugins[gid] = plugin

    def get(self, game_id: str) -> Optional[GamePlugin]:
        gid = (game_id or "").strip().lower()
        return self._plugins.get(gid)

    def list_games(self) -> List[GameInfo]:
        return [p.info() for p in self._plugins.values()]

    def ids(self) -> List[str]:
        return list(self._plugins.keys())


