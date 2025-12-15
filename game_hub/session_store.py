"""
In-memory session store for GameHub.

Session scope: Discord voice channel ID (string).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SessionState:
    active_game_id: Optional[str] = None


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState:
        sid = str(session_id)
        if sid not in self._sessions:
            self._sessions[sid] = SessionState(active_game_id=None)
        return self._sessions[sid]

    def start(self, session_id: str, game_id: str) -> SessionState:
        st = self.get(session_id)
        st.active_game_id = (game_id or "").strip().lower() or None
        return st

    def stop(self, session_id: str) -> SessionState:
        st = self.get(session_id)
        st.active_game_id = None
        return st


