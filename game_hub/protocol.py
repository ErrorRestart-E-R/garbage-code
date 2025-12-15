"""
GameHub HTTP API protocol (Pydantic models).

The GameHub does NOT call the main LLM (by design).
It returns a prompt patch that the main core injects into its own LLM call.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional


class GameInfo(BaseModel):
    id: str = Field(..., description="Stable game ID (ascii slug)")
    name: str = Field(..., description="Display name (Korean/English)")
    aliases: List[str] = Field(default_factory=list, description="User utterance aliases")
    description: str = Field("", description="Short description")


class GamesResponse(BaseModel):
    games: List[GameInfo]


class StartGameRequest(BaseModel):
    game_id: str


class StartGameResponse(BaseModel):
    active_game_id: Optional[str] = None


class StopGameResponse(BaseModel):
    active_game_id: Optional[str] = None


class PrepareTurnRequest(BaseModel):
    last_user_text: str
    recent_turns: List[str] = Field(default_factory=list, description="Recent user/assistant turns, already formatted")
    active_game_id: Optional[str] = None


class ContextBlock(BaseModel):
    title: str = ""
    content: str


class PrepareTurnResponse(BaseModel):
    active_game_id: Optional[str] = None
    system_addendum: str = ""
    context_blocks: List[ContextBlock] = Field(default_factory=list)
    allowed_controls: List[str] = Field(default_factory=list, description="e.g., ['NONE','STOP','START:ktane']")
    trace_id: str = ""


