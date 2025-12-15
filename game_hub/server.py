"""
GameHub HTTP server (FastAPI).

Run as a separate process. The main bot calls this service to get prompt patches.
"""

from __future__ import annotations

import os
import uuid
from fastapi import FastAPI, HTTPException

from .protocol import (
    GamesResponse,
    PrepareTurnRequest,
    PrepareTurnResponse,
    StartGameRequest,
    StartGameResponse,
    StopGameResponse,
)
from .registry import GameRegistry
from .session_store import SessionStore
from .games.ktane.plugin import KtanePlugin


def create_app() -> FastAPI:
    app = FastAPI(title="AiVutber GameHub", version="0.1.0")

    registry = GameRegistry()
    sessions = SessionStore()

    # Register built-in games (plugins)
    registry.register(KtanePlugin())

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/v1/games", response_model=GamesResponse)
    def list_games() -> GamesResponse:
        return GamesResponse(games=registry.list_games())

    @app.post("/v1/sessions/{session_id}/start", response_model=StartGameResponse)
    def start_game(session_id: str, body: StartGameRequest) -> StartGameResponse:
        gid = (body.game_id or "").strip().lower()
        if not gid:
            raise HTTPException(status_code=400, detail="game_id is empty")
        if registry.get(gid) is None:
            raise HTTPException(status_code=404, detail=f"unknown game_id: {gid}")
        st = sessions.start(session_id, gid)
        return StartGameResponse(active_game_id=st.active_game_id)

    @app.post("/v1/sessions/{session_id}/stop", response_model=StopGameResponse)
    def stop_game(session_id: str) -> StopGameResponse:
        st = sessions.stop(session_id)
        return StopGameResponse(active_game_id=st.active_game_id)

    @app.post("/v1/sessions/{session_id}/prepare_turn", response_model=PrepareTurnResponse)
    def prepare_turn(session_id: str, body: PrepareTurnRequest) -> PrepareTurnResponse:
        trace_id = uuid.uuid4().hex

        # Determine active game: prefer hub session store; fall back to provided.
        st = sessions.get(session_id)
        if body.active_game_id and not st.active_game_id:
            st.active_game_id = (body.active_game_id or "").strip().lower() or None

        active = st.active_game_id
        if not active:
            # No active game => no patch. Main core may still use this to show game list.
            return PrepareTurnResponse(
                active_game_id=None,
                system_addendum="",
                context_blocks=[],
                allowed_controls=["NONE"],
                trace_id=trace_id,
            )

        plugin = registry.get(active)
        if plugin is None:
            # stale session state
            st.active_game_id = None
            return PrepareTurnResponse(
                active_game_id=None,
                system_addendum="",
                context_blocks=[],
                allowed_controls=["NONE"],
                trace_id=trace_id,
            )

        res = plugin.prepare_turn(
            session_id=str(session_id),
            last_user_text=body.last_user_text,
            recent_turns=body.recent_turns or [],
        )
        return PrepareTurnResponse(
            active_game_id=active,
            system_addendum=res.system_addendum,
            context_blocks=res.context_blocks,
            allowed_controls=res.allowed_controls,
            trace_id=trace_id,
        )

    return app


# Default ASGI app for uvicorn: `uvicorn game_hub.server:app --port 8765`
app = create_app()


if __name__ == "__main__":
    # Convenience: python game_hub/server.py
    import uvicorn

    host = os.getenv("GAME_HUB_HOST", "127.0.0.1")
    port = int(os.getenv("GAME_HUB_PORT", "8765"))
    uvicorn.run("game_hub.server:app", host=host, port=port, reload=False)


