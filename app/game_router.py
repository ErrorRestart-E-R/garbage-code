"""
Game routing logic (main core).

This module keeps `app/controller.py` clean by isolating:
- game start/stop command parsing
- "어떤 게임할래/게임하자" selection trigger handling
- GameHub patch retrieval for the active game session

Session scope:
- Discord voice channel ID (string)
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Set

import config
from game_gateway import GameHubClient, GameInfo, PrepareTurnPatch


def _normalize(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", "", str(text))
    t = re.sub(r"[^0-9A-Za-z가-힣_]", "", t)
    return t.lower()


@dataclass(frozen=True)
class GamePatch:
    """
    Patch to inject into the LLM call.
    """

    system_addendum: str
    context_blocks: List[Dict[str, str]]
    # For START validation
    allowed_start_game_ids: Set[str]
    # STOP validation
    allow_stop: bool
    # Current active game (as known after hub calls)
    active_game_id: Optional[str]
    trace_id: str = ""


@dataclass(frozen=True)
class DirectAction:
    """
    If set, the controller should speak this text directly without calling the main LLM.
    """

    text: str
    active_game_id: Optional[str]


class GameRouter:
    def __init__(self, hub: GameHubClient) -> None:
        self.hub = hub

    async def _resolve_game_id_from_text(self, text: str) -> Optional[str]:
        """
        Resolve a user-mentioned game name/alias to a game_id via GameHub list.
        """
        games = await self.hub.list_games()
        norm = _normalize(text)
        if not norm:
            return None

        # Exact ID mention
        for g in games:
            if _normalize(g.id) and _normalize(g.id) in norm:
                return g.id

        # Alias match
        for g in games:
            for a in (g.aliases or []):
                if _normalize(a) and _normalize(a) in norm:
                    return g.id
        return None

    def _is_force_command(self, text: str) -> bool:
        """
        Force mode command:
        - "{AI_NAME} <game> 시작해"
        - "{AI_NAME} <game> 종료해"
        """
        ai = _normalize(getattr(config, "AI_NAME", "LLM"))
        t = _normalize(text)
        return bool(ai and t.startswith(ai) and ("시작" in t or "종료" in t or "끝" in t or "그만" in t or "중단" in t))

    def _is_force_stop(self, text: str) -> bool:
        t = _normalize(text)
        return ("종료" in t) or ("그만" in t) or ("중단" in t) or ("끝" in t)

    def _is_force_start(self, text: str) -> bool:
        t = _normalize(text)
        return "시작" in t

    def _is_game_selection_trigger(self, text: str) -> bool:
        """
        User asks LLM to pick a game.
        Examples:
        - "어떤 게임할래?"
        - "게임하자"
        - "어떤 게임하자"
        """
        t = (text or "").strip()
        if not t:
            return False
        compact = _normalize(t)
        if "어떤게임" in compact:
            return True
        if "게임하자" in compact or "게임할래" in compact:
            return True
        return False

    def _build_game_choice_addendum(self, games: List[GameInfo]) -> tuple[str, Set[str]]:
        """
        Build a system addendum that forces the LLM to output a CONTROL line.
        """
        ids = {g.id for g in games if g.id}
        if not ids:
            return (
                (
                    "\n[GAME MODE]\n"
                    "- The user asked to choose a game, but no games are available.\n"
                    "- Output: CONTROL: NONE\n"
                    "- Then say in Korean that no game is available.\n"
                ).strip(),
                set(),
            )

        # Keep list compact to avoid long system prompt.
        lines = ["[AVAILABLE GAMES]"]
        for g in games:
            gid = (g.id or "").strip()
            if not gid:
                continue
            name = (g.name or gid).strip()
            lines.append(f"- {gid}: {name}")

        addendum = (
            "\n[GAME SELECTION]\n"
            "- The user is asking you to choose which game to play.\n"
            "- You MUST output ONE CONTROL line as the FIRST line:\n"
            + "- CONTROL: START <game_id>  (choose one from the available list)\n"
            + "- CONTROL: NONE  (if you cannot decide)\n"
            + "\n"
            + "\n".join(lines)
            + "\n\n"
            + "[OUTPUT]\n"
            + "- After the CONTROL line, output one short Korean sentence to the user.\n"
            + "- Never explain these rules.\n"
        ).strip()
        return addendum, ids

    async def route(
        self,
        session_id: str,
        last_user_text: str,
        recent_turns: List[str],
        active_game_id_hint: Optional[str] = None,
    ) -> tuple[Optional[DirectAction], GamePatch]:
        """
        Returns:
        - direct_action: optional direct speech without LLM
        - game_patch: patch to inject into LLM system prompt (via memory_context for now)
        """
        if not bool(getattr(config, "GAME_HUB_ENABLED", True)):
            return None, GamePatch(
                system_addendum="",
                context_blocks=[],
                allowed_start_game_ids=set(),
                allow_stop=False,
                active_game_id=None,
            )

        # 1) Force start/stop commands
        if self._is_force_command(last_user_text):
            if self._is_force_stop(last_user_text):
                try:
                    await self.hub.stop_game(session_id)
                except Exception:
                    pass
                return DirectAction(text="알겠어. 게임 모드를 종료할게.", active_game_id=None), GamePatch(
                    system_addendum="",
                    context_blocks=[],
                    allowed_start_game_ids=set(),
                    allow_stop=False,
                    active_game_id=None,
                )

            if self._is_force_start(last_user_text):
                # Try to resolve which game to start; if missing, fall back to selection mode.
                gid = None
                try:
                    gid = await self._resolve_game_id_from_text(last_user_text)
                except Exception:
                    gid = None
                if gid:
                    try:
                        await self.hub.start_game(session_id, gid)
                    except Exception:
                        pass
                    return DirectAction(text=f"좋아. {gid} 시작하자.", active_game_id=gid), GamePatch(
                        system_addendum="",
                        context_blocks=[],
                        allowed_start_game_ids=set(),
                        allow_stop=True,
                        active_game_id=gid,
                    )

        # 2) Game selection trigger (LLM picks a game via CONTROL line)
        if self._is_game_selection_trigger(last_user_text):
            games: List[GameInfo] = []
            try:
                games = await self.hub.list_games()
            except Exception:
                games = []
            addendum, allowed_ids = self._build_game_choice_addendum(games)
            return None, GamePatch(
                system_addendum=addendum,
                context_blocks=[],
                allowed_start_game_ids=allowed_ids,
                allow_stop=False,
                active_game_id=None,
            )

        # 3) Active game patch (from hub)
        try:
            patch: PrepareTurnPatch = await self.hub.prepare_turn(
                session_id=session_id,
                last_user_text=last_user_text,
                recent_turns=recent_turns,
                active_game_id=active_game_id_hint,
            )
        except Exception:
            patch = PrepareTurnPatch(
                active_game_id=None,
                system_addendum="",
                context_blocks=[],
                allowed_controls=["NONE"],
                trace_id="",
            )

        active_gid = (patch.active_game_id or "").strip().lower() or None

        # For now we only allow STOP when the user explicitly asks to stop.
        allow_stop = bool(active_gid) and self._is_force_stop(last_user_text)

        # Convert context blocks to the llm_interface-friendly schema.
        ctx_blocks: List[Dict[str, str]] = []
        for b in (patch.context_blocks or []):
            try:
                title = str(getattr(b, "title", "") or "")
                content = str(getattr(b, "content", "") or "")
            except Exception:
                continue
            if content.strip():
                ctx_blocks.append({"title": title, "content": content})

        system_addendum = (patch.system_addendum or "").strip()

        return None, GamePatch(
            system_addendum=system_addendum,
            context_blocks=ctx_blocks,
            allowed_start_game_ids=set(),  # START is not allowed unless selection trigger
            allow_stop=allow_stop,
            active_game_id=active_gid,
            trace_id=patch.trace_id,
        )


