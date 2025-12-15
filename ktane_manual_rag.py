"""
KTANE Manual RAG (text-only)

목표:
- 사용자가 PDF를 텍스트로 변환해 둔 "로컬 매뉴얼 텍스트"를 기반으로
  유저 발화(예: "6개의 와이어가 있는데...")와 가장 관련 있는 부분을 찾아
  LLM 시스템 프롬프트에 주입하기 위한 경량 RAG 모듈.

설계 원칙:
- 게임 모드 OFF일 때는 절대 로드/실행되지 않도록(성능/부작용 최소화)
- import 시점에 무거운 모델 다운로드/로딩을 하지 않도록(지연 로딩)
- 파일이 없거나 임베딩 모델 로딩이 실패해도 "키워드 기반" 폴백으로 동작
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import threading
from typing import List, Optional, Sequence, Tuple

import numpy as np
import requests

from logger import setup_logger
import config


logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)


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
    Tries UTF-8 first, then falls back to cp949.
    """
    if not path:
        return ""
    # normalize path for metadata; reading with original is fine
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="cp949") as f:
                return f.read()
        except Exception:
            return ""
    except Exception:
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
            # Remove previously buffered prev_line if it is there
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
        self.ollama_base_url = (
            (ollama_base_url or "").strip()
            or getattr(config, "OLLAMA_EMBEDDING_URL", "").strip()
            or getattr(config, "OLLAMA_LLM_URL", "").strip()
            or "http://localhost:11434"
        )
        self.ollama_timeout_seconds = float(ollama_timeout_seconds)
        self.ollama_batch_size = max(1, int(ollama_batch_size))
        self.top_k = int(top_k) if top_k else 4
        self.chunk_size_chars = int(chunk_size_chars) if chunk_size_chars else 1200
        self.overlap_chars = int(overlap_chars) if overlap_chars else 150
        self.min_score = float(min_score)

        self._lock = threading.Lock()
        self._model = None  # SentenceTransformer (lazy)
        self._chunks: List[Tuple[str, str]] = []  # (source, text)
        self._emb: Optional[np.ndarray] = None  # shape: (N, D), normalized
        self._fingerprint: Optional[Tuple] = None

    @staticmethod
    def _fingerprint_paths(paths: Sequence[str]) -> Tuple:
        fp = []
        for p in paths:
            try:
                st = os.stat(p)
                fp.append((os.path.abspath(p), int(st.st_mtime), int(st.st_size)))
            except Exception:
                fp.append((os.path.abspath(p), None, None))
        return tuple(fp)

    def _load_sentence_transformer(self):
        if self._model is not None:
            return self._model
        # Lazy import to avoid heavy startup when not needed
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(self.embedding_model)
        return self._model

    def _resolve_provider(self) -> str:
        """
        Decide which embedding backend to use.
        - sentence_transformers: local embeddings via HF models
        - ollama: remote embeddings via Ollama HTTP API
        """
        p = (self.embedding_provider or "auto").strip().lower()
        if p in ("sentence_transformers", "sentence-transformers", "sbert", "local"):
            return "sentence_transformers"
        if p in ("ollama", "ollama_api", "ollama-http"):
            return "ollama"
        # auto
        m = (self.embedding_model or "").strip()
        # Heuristic:
        # - Ollama model names commonly include a tag like ":latest"
        # - SentenceTransformer model IDs usually don't include ":" (except rare local paths)
        if ":" in m and not re.match(r"^[A-Za-z]:[\\/]", m):
            return "ollama"
        return "sentence_transformers"

    def _ollama_post(self, endpoint: str, payload: dict) -> dict:
        base = (self.ollama_base_url or "").rstrip("/")
        url = f"{base}{endpoint}"
        r = requests.post(url, json=payload, timeout=self.ollama_timeout_seconds)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {}

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        provider = self._resolve_provider()
        if provider == "ollama":
            return self._embed_texts_ollama(texts)
        return self._embed_texts_sentence_transformers(texts)

    def _embed_query(self, text: str) -> np.ndarray:
        provider = self._resolve_provider()
        if provider == "ollama":
            return self._embed_query_ollama(text)
        return self._embed_query_sentence_transformers(text)

    def _embed_texts_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        model = self._load_sentence_transformer()
        emb = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _l2_normalize(np.asarray(emb, dtype=np.float32))

    def _embed_query_sentence_transformers(self, text: str) -> np.ndarray:
        model = self._load_sentence_transformer()
        q_emb = model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return _l2_normalize(np.asarray(q_emb, dtype=np.float32))

    def _embed_texts_ollama(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts using Ollama.
        Tries /api/embed (batch) first, falls back to /api/embeddings (single) if needed.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        model_name = (self.embedding_model or "").strip()
        if not model_name:
            raise ValueError("KTANE_EMBEDDING_MODEL is empty for ollama provider")

        all_rows: List[np.ndarray] = []
        for i in range(0, len(texts), self.ollama_batch_size):
            batch = texts[i : i + self.ollama_batch_size]
            rows = self._ollama_embed_batch(batch)
            all_rows.append(rows)

        mat = np.vstack(all_rows) if all_rows else np.zeros((0, 0), dtype=np.float32)
        return _l2_normalize(mat)

    def _embed_query_ollama(self, text: str) -> np.ndarray:
        rows = self._ollama_embed_batch([text])
        if rows.ndim == 2 and rows.shape[0] >= 1:
            return _l2_normalize(rows[0])
        return _l2_normalize(np.asarray(rows, dtype=np.float32))

    def _ollama_embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Best-effort Ollama embedding for multiple inputs.
        """
        model_name = (self.embedding_model or "").strip()

        # 1) Try modern batch endpoint: /api/embed
        try:
            data = self._ollama_post("/api/embed", {"model": model_name, "input": texts})
            embs = data.get("embeddings")
            if isinstance(embs, list) and embs and isinstance(embs[0], list):
                return np.asarray(embs, dtype=np.float32)
        except requests.HTTPError as e:
            # 404/400 likely means endpoint not supported / schema mismatch
            if getattr(e.response, "status_code", None) not in (400, 404):
                raise
        except Exception:
            pass

        # 2) Fallback: /api/embeddings (single prompt)
        rows: List[List[float]] = []
        for t in texts:
            # try prompt schema
            data = None
            try:
                data = self._ollama_post("/api/embeddings", {"model": model_name, "prompt": t})
            except requests.HTTPError as e:
                if getattr(e.response, "status_code", None) == 400:
                    # try alternative schema
                    data = self._ollama_post("/api/embeddings", {"model": model_name, "input": t})
                else:
                    raise
            emb = (data or {}).get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError("Ollama embeddings response missing 'embedding' list")
            rows.append(emb)

        return np.asarray(rows, dtype=np.float32)

    def _build_index_locked(self) -> Optional[str]:
        """
        Build chunk list + embeddings. Must be called under lock.
        Returns error message if any.
        """
        if not self.manual_paths:
            self._chunks = []
            self._emb = None
            self._fingerprint = self._fingerprint_paths([])
            return "KTANE_MANUAL_TEXT_PATHS가 비어있습니다."

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
            return "KTANE 매뉴얼 텍스트를 읽었지만 청크가 0개입니다. 파일 경로/인코딩을 확인하세요."

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
            return f"임베딩 로딩 실패(키워드 검색으로 폴백): {e}"

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
            return RagResult(hits=[], top_score=0.0, error=err or "KTANE 매뉴얼 인덱스가 비어있습니다.")

        k = int(top_k) if top_k else self.top_k
        k = max(1, min(k, 10))

        # Embedding mode
        if emb is not None:
            try:
                q_emb = self._embed_query(q)
                # cosine similarity between normalized vectors is dot product
                sims = emb @ q_emb
                # Get top-k indices
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
                    # fallback to keyword search
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

