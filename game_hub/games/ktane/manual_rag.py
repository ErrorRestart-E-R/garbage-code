"""
KTANE Manual RAG (text-only) for GameHub.

This file is migrated from the main repo-level `ktane_manual_rag.py` so that
KTANE-specific logic lives under `game_hub/` (separate service).

Design principles:
- Lazy loading: do not download/load heavy models at import time
- Robust fallback: if embeddings fail, use keyword-based retrieval
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re
import threading
from typing import List, Optional, Sequence, Tuple

import numpy as np
import requests


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagHit:
    text: str
    source: str
    score: float


@dataclass(frozen=True)
class RagResult:
    hits: List[RagHit]
    top_score: float
    error: Optional[str] = None


def _read_text_file(path: str) -> str:
    """
    Best-effort text file reader for Windows/Korean environments.
    Prefer UTF-8. Also supports UTF-8 BOM and UTF-16.
    """
    if not path:
        return ""
    encodings = ("utf-8-sig", "utf-8", "utf-16", "cp949")
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            if text and enc not in ("utf-8-sig", "utf-8"):
                logger.warning(
                    f"[KTANE] manual text loaded with fallback encoding={enc}. Please save as UTF-8: {os.path.abspath(path)}"
                )
            return text
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    return ""


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs while preserving headings as their own paragraphs.
    """
    if not text:
        return []

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    paras: List[str] = []
    buf: List[str] = []

    def flush():
        nonlocal buf
        if buf:
            p = "\n".join(buf).strip()
            if p:
                paras.append(p)
            buf = []

    heading_re = re.compile(r"^\s{0,3}#{1,6}\s+\S+")
    underline_heading_re = re.compile(r"^\s*[-=]{3,}\s*$")

    prev_line = ""
    for ln in lines:
        raw = ln.rstrip("\n")
        stripped = raw.strip()

        # blank line => paragraph boundary
        if not stripped:
            flush()
            prev_line = raw
            continue

        # markdown heading
        if heading_re.match(raw):
            flush()
            paras.append(stripped)
            prev_line = raw
            continue

        # underline style heading (prev line is title, current is ===== / -----)
        if underline_heading_re.match(raw) and prev_line and prev_line.strip():
            # Convert "Title\n=====" into one paragraph for chunking
            flush()
            paras.append(prev_line.strip())
            prev_line = raw
            continue

        buf.append(raw)
        prev_line = raw

    flush()
    return paras


def _chunk_paragraphs(
    paragraphs: Sequence[str],
    chunk_size_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[str]:
    """
    Create overlapping chunks from paragraph list.
    - chunk_size_chars: approximate maximum characters per chunk
    - overlap_chars: carry over last N characters to next chunk for continuity
    """
    if not paragraphs:
        return []

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return
        txt = "\n\n".join(cur).strip()
        if txt:
            chunks.append(txt)
        # prepare overlap buffer
        if overlap_chars > 0 and txt:
            tail = txt[-overlap_chars:]
            cur = [tail]
            cur_len = len(tail)
        else:
            cur = []
            cur_len = 0

    for p in paragraphs:
        p = (p or "").strip()
        if not p:
            continue

        # If a single paragraph is huge, split it hard.
        if len(p) > chunk_size_chars * 2:
            # flush current first
            flush()
            for i in range(0, len(p), chunk_size_chars):
                part = p[i : i + chunk_size_chars].strip()
                if part:
                    chunks.append(part)
            cur = []
            cur_len = 0
            continue

        add_len = len(p) + (2 if cur else 0)
        if cur_len + add_len > chunk_size_chars and cur:
            flush()

        cur.append(p)
        cur_len += add_len

    # final
    if cur:
        # don't create extra overlap on last flush
        txt = "\n\n".join(cur).strip()
        if txt:
            chunks.append(txt)

    # remove duplicates that can happen due to overlap-only chunks
    uniq: List[str] = []
    seen = set()
    for c in chunks:
        key = c.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    return uniq


def _keyword_score(query: str, doc: str) -> float:
    """
    Simple lexical overlap score as a fallback (0..1-ish).
    """
    q = (query or "").strip().lower()
    d = (doc or "").strip().lower()
    if not q or not d:
        return 0.0

    # Tokenize by Korean/English/numbers chunks
    tokens = re.findall(r"[0-9A-Za-z가-힣]{2,}", q)
    if not tokens:
        return 0.0

    hits = sum(1 for t in tokens if t in d)
    return hits / max(3.0, float(len(tokens)))


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors for cosine similarity via dot product.
    """
    if x is None:
        return x
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        if norm <= 0:
            return arr
        return arr / norm
    if arr.ndim == 2:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
    return arr


class KtaneManualRag:
    """
    Local-text manual retriever.
    """

    def __init__(
        self,
        manual_paths: Sequence[str],
        embedding_model: str,
        embedding_provider: str = "auto",
        ollama_base_url: Optional[str] = None,
        ollama_timeout_seconds: float = 30.0,
        ollama_batch_size: int = 32,
        top_k: int = 4,
        chunk_size_chars: int = 1200,
        overlap_chars: int = 150,
        min_score: float = 0.15,
    ) -> None:
        self.manual_paths = [p for p in (manual_paths or []) if isinstance(p, str) and p.strip()]
        self.embedding_model = embedding_model
        self.embedding_provider = (embedding_provider or "auto").strip()
        self.ollama_base_url = (ollama_base_url or "").strip()
        self.ollama_timeout_seconds = float(ollama_timeout_seconds or 30.0)
        self.ollama_batch_size = int(ollama_batch_size or 32)
        self.top_k = int(top_k or 4)
        self.chunk_size_chars = int(chunk_size_chars or 1200)
        self.overlap_chars = int(overlap_chars or 150)
        self.min_score = float(min_score or 0.15)

        self._lock = threading.Lock()
        self._fingerprint = ""
        self._chunks: List[Tuple[str, str]] = []  # (source, chunk_text)
        self._emb: Optional[np.ndarray] = None  # (N, dim) normalized

    @staticmethod
    def _fingerprint_paths(paths: Sequence[str]) -> str:
        parts: List[str] = []
        for p in paths or []:
            try:
                st = os.stat(p)
                parts.append(f"{os.path.abspath(p)}:{st.st_mtime}:{st.st_size}")
            except Exception:
                parts.append(f"{os.path.abspath(p)}:missing")
        return "|".join(parts)

    def _resolve_provider(self) -> str:
        p = (self.embedding_provider or "auto").strip().lower()
        if p in ("ollama", "sentence_transformers"):
            return p
        # auto
        m = (self.embedding_model or "").strip().lower()
        if ":" in m or "ollama" in m:
            return "ollama"
        return "sentence_transformers"

    def _embed_texts_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        model_name = (self.embedding_model or "").strip()
        if not model_name:
            raise ValueError("embedding_model is empty for sentence_transformers provider")

        model = SentenceTransformer(model_name)
        vecs = model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    def _embed_texts_ollama(self, texts: List[str]) -> np.ndarray:
        base = (self.ollama_base_url or "").strip()
        if not base:
            raise ValueError("ollama_base_url is empty for ollama provider")

        model_name = (self.embedding_model or "").strip()
        if not model_name:
            raise ValueError("embedding_model is empty for ollama provider")

        url = base.rstrip("/") + "/api/embeddings"
        timeout = float(self.ollama_timeout_seconds or 30.0)
        batch = max(1, min(int(self.ollama_batch_size or 32), 128))

        rows: List[List[float]] = []
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            payload = {"model": model_name, "prompt": chunk}
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()

            # Ollama embeddings API can return:
            # - {"embedding": [...] } for single prompt
            # - {"embeddings": [{"embedding": [...]}, ...]} for batch (depending on version)
            if "embedding" in data:
                emb = data.get("embedding")
                if not isinstance(emb, list):
                    raise RuntimeError("Ollama embeddings response missing 'embedding' list")
                rows.append(emb)
                continue

            embs = data.get("embeddings")
            if not isinstance(embs, list):
                raise RuntimeError("Ollama embeddings response missing 'embeddings' list")
            for item in embs:
                emb = item.get("embedding") if isinstance(item, dict) else None
                if not isinstance(emb, list):
                    raise RuntimeError("Ollama embeddings response item missing 'embedding' list")
                rows.append(emb)

        return np.asarray(rows, dtype=np.float32)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        provider = self._resolve_provider()
        if provider == "ollama":
            vecs = self._embed_texts_ollama(texts)
        else:
            vecs = self._embed_texts_sentence_transformers(texts)
        vecs = _l2_normalize(vecs)
        return vecs

    def _embed_query(self, q: str) -> np.ndarray:
        vec = self._embed_texts([q])
        if vec.ndim == 2 and vec.shape[0] == 1:
            return vec[0]
        return vec

    def _build_index_locked(self) -> Optional[str]:
        """
        Build chunk list + embeddings. Must be called under lock.
        Returns error message if any.
        """
        if not self.manual_paths:
            self._chunks = []
            self._emb = None
            self._fingerprint = self._fingerprint_paths([])
            return "KTANE_MANUAL_TEXT_PATHS is empty."

        fp = self._fingerprint_paths(self.manual_paths)
        if self._fingerprint == fp and self._chunks and self._emb is not None:
            return None  # up to date

        chunks: List[Tuple[str, str]] = []
        for path in self.manual_paths:
            abs_path = os.path.abspath(path)
            if not os.path.exists(path):
                logger.warning(f"[KTANE] manual text not found: {abs_path}")
                continue

            raw = _read_text_file(path)
            if not raw.strip():
                logger.warning(f"[KTANE] manual text is empty/unreadable: {abs_path}")
                continue

            paras = _split_into_paragraphs(raw)
            file_chunks = _chunk_paragraphs(
                paras,
                chunk_size_chars=self.chunk_size_chars,
                overlap_chars=self.overlap_chars,
            )
            for c in file_chunks:
                chunks.append((os.path.basename(abs_path), c))

        if not chunks:
            self._chunks = []
            self._emb = None
            self._fingerprint = fp
            return "KTANE manual chunks are empty."

        self._chunks = chunks
        self._fingerprint = fp

        # Try embedding; if it fails, keep only keyword mode.
        try:
            texts = [c for _, c in self._chunks]
            self._emb = self._embed_texts(texts)
            dim = int(self._emb.shape[1]) if self._emb.ndim == 2 else 0
            provider = self._resolve_provider()
            logger.info(f"[KTANE] manual index built: chunks={len(self._chunks)} dim={dim} provider={provider}")
            return None
        except Exception as e:
            self._emb = None
            logger.warning(f"[KTANE] embedding model unavailable. fallback to keyword search only. err={e}")
            return f"embedding unavailable (keyword fallback): {e}"

    def ensure_index(self) -> Optional[str]:
        with self._lock:
            return self._build_index_locked()

    def query(self, query_text: str, top_k: Optional[int] = None) -> RagResult:
        q = (query_text or "").strip()
        if not q:
            return RagResult(hits=[], top_score=0.0, error=None)

        err = self.ensure_index()

        with self._lock:
            chunks = list(self._chunks)
            emb = self._emb

        if not chunks:
            return RagResult(hits=[], top_score=0.0, error=err or "KTANE index is empty.")

        k = int(top_k) if top_k else self.top_k
        k = max(1, min(k, 10))

        # Embedding mode
        if emb is not None:
            try:
                q_emb = self._embed_query(q)
                sims = emb @ q_emb
                idx = np.argsort(-sims)[:k]
                hits: List[RagHit] = []
                top_score = float(sims[idx[0]]) if len(idx) > 0 else 0.0
                for i in idx:
                    s = float(sims[i])
                    if s < self.min_score:
                        continue
                    src, txt = chunks[int(i)]
                    hits.append(RagHit(text=txt, source=src, score=s))
                if not hits:
                    return self._query_keyword_fallback(q, chunks, k, err)
                return RagResult(hits=hits, top_score=top_score, error=err)
            except Exception as e:
                logger.warning(f"[KTANE] embedding query failed; fallback to keyword search. err={e}")
                return self._query_keyword_fallback(q, chunks, k, err or str(e))

        # Keyword-only mode
        return self._query_keyword_fallback(q, chunks, k, err)

    def _query_keyword_fallback(
        self,
        q: str,
        chunks: Sequence[Tuple[str, str]],
        k: int,
        err: Optional[str],
    ) -> RagResult:
        scored: List[Tuple[float, int]] = []
        for i, (_, txt) in enumerate(chunks):
            s = _keyword_score(q, txt)
            scored.append((s, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]
        hits: List[RagHit] = []
        top_score = float(top[0][0]) if top else 0.0
        for s, idx in top:
            if s <= 0:
                continue
            src, txt = chunks[idx]
            hits.append(RagHit(text=txt, source=src, score=float(s)))
        return RagResult(hits=hits, top_score=top_score, error=err)

    @staticmethod
    def format_context(result: RagResult, max_chars: int = 6000, max_chars_per_hit: int = 900) -> str:
        """
        Turn retrieved hits into a single prompt-ready context string.
        """
        if not result or not result.hits:
            return ""

        budget = max(500, int(max_chars))
        per_hit = max(200, int(max_chars_per_hit))
        parts: List[str] = []
        used = 0
        for h in result.hits:
            header = f"[source={h.source} score={h.score:.3f}]"
            text = (h.text or "").strip()
            if len(text) > per_hit:
                text = text[:per_hit].rstrip()
            block = f"{header}\n{text}".strip()
            if not block:
                continue
            if used + len(block) + 2 > budget and parts:
                break
            parts.append(block)
            used += len(block) + 2
        return "\n\n---\n\n".join(parts).strip()


