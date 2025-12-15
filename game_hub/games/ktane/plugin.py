"""
KTANE plugin for GameHub.

Responsibilities:
- Provide game metadata (id/aliases)
- Build a prompt patch for the main LLM: system_addendum + retrieved manual context
- Keep the patch small and deterministic
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from ...protocol import ContextBlock, GameInfo
from ...registry import PrepareTurnResult
from .manual_rag import KtaneManualRag


class KtanePlugin:
    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent
        manual_path = base_dir / "ktane_manual.txt"

        embedding_provider = os.getenv("KTANE_EMBEDDING_PROVIDER", "auto")
        embedding_model = os.getenv("KTANE_EMBEDDING_MODEL", "embeddinggemma:latest")
        ollama_base_url = os.getenv("OLLAMA_EMBEDDING_URL", "")  # reuse existing env name if present

        top_k = int(os.getenv("KTANE_RAG_TOP_K", "4"))
        self._max_ctx_chars = int(os.getenv("KTANE_RAG_MAX_CONTEXT_CHARS", "4000"))

        self._rag = KtaneManualRag(
            manual_paths=[str(manual_path)],
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            ollama_base_url=ollama_base_url or None,
            top_k=top_k,
        )

    def info(self) -> GameInfo:
        return GameInfo(
            id="ktane",
            name="KTANE(폭탄 해체)",
            aliases=[
                "ktane",
                "keep talking",
                "nobody explodes",
                "폭탄",
                "폭탄해체",
                "폭탄 해체",
            ],
            description="Keep Talking and Nobody Explodes 폭탄 해체 가이드",
        )

    def prepare_turn(self, session_id: str, last_user_text: str, recent_turns: List[str]) -> PrepareTurnResult:
        # Broaden query slightly using recent turns (avoids cases where the latest utterance is too short).
        query_parts: List[str] = []
        if recent_turns:
            # Keep last few turns only to avoid noise
            tail = recent_turns[-4:]
            query_parts.extend([t for t in tail if isinstance(t, str) and t.strip()])
        if last_user_text and last_user_text.strip():
            query_parts.append(last_user_text.strip())
        query_text = "\n".join(query_parts).strip()

        rag_result = self._rag.query(query_text)
        manual_ctx = self._rag.format_context(rag_result, max_chars=self._max_ctx_chars)

        system_addendum = (
            "\n[GAME MODE: KTANE]\n"
            "- You are assisting with 'KEEP TALKING and NOBODY EXPLODES'.\n"
            "- Use only the provided manual context blocks for rules.\n"
            "- If one crucial fact is missing, ask only 1 short question.\n"
            "- Do not mention manuals/contexts/RAG.\n"
            "- Output only the immediate action(s) now, in Korean, in one short line.\n"
            "- For Keypads: never invent symbols; press only the 4 given symbols in column order.\n"
            "- For Wires: output only which wire number to cut.\n"
            "- For Button: output only tap vs hold; if hold, include the release timing rule.\n"
        ).strip()

        blocks: List[ContextBlock] = []
        if manual_ctx:
            blocks.append(ContextBlock(title="KTANE manual context", content=manual_ctx))

        # While in KTANE, allow STOP (user may want to end the game), or NONE.
        allowed_controls = ["NONE", "STOP"]

        return PrepareTurnResult(
            system_addendum=system_addendum,
            context_blocks=blocks,
            allowed_controls=allowed_controls,
        )


