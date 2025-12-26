"""
Memory Manager: Mem0 with Ollama (Separate from Main LLM)

Uses Mem0 for intelligent memory management:
- Search: Ollama embedding model only (no LLM, fast)
- Save: Ollama small LLM for fact extraction (runs in separate process)

This keeps memory operations completely separate from the main llama.cpp LLM.
"""

from mem0 import Memory
import config
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Optional
from logger import setup_logger
import re
import time
import os

logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)

# Process pool for memory operations (avoids blocking main thread)
_memory_executor: Optional[ProcessPoolExecutor] = None


def _init_memory_in_process():
    """Initialize Mem0 in a separate process"""
    global _process_memory
    _process_memory = Memory.from_config(config.MEM0_CONFIG)
    return True


def _save_memory_process(user_name: str, text: str) -> Optional[Dict]:
    """
    Save memory in a separate process.
    This function runs in ProcessPoolExecutor to avoid blocking.
    """
    try:
        # Initialize memory if not already done in this process
        memory = Memory.from_config(config.MEM0_CONFIG)
        
        result = memory.add(
            text,
            user_id=user_name,
            metadata={"source": "voice_chat"}
        )
        
        if result and result.get("results"):
            return {"success": True, "result": result}
        return {"success": True, "result": None}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


class MemoryManager:
    def __init__(self):
        """
        Initialize Mem0 Memory with Ollama configuration.
        Uses separate Ollama models for embedding and LLM (not llama.cpp).
        """
        self.enabled = getattr(config, 'ENABLE_MEMORY', True)
        
        if not self.enabled:
            self.memory = None
            logger.info("Mem0 Memory is DISABLED")
            return
        
        try:
            # If mem0 is configured to use llama.cpp via OpenAI-compatible provider (lmstudio),
            # some clients require OPENAI_API_KEY to be set even if the server doesn't enforce it.
            try:
                llm_cfg = getattr(config, "MEM0_CONFIG", {}).get("llm", {}) if isinstance(getattr(config, "MEM0_CONFIG", None), dict) else {}
                provider = (llm_cfg.get("provider") or "").strip().lower() if isinstance(llm_cfg, dict) else ""
                if provider == "lmstudio":
                    if not (os.getenv("OPENAI_API_KEY") or "").strip():
                        os.environ["OPENAI_API_KEY"] = str(getattr(config, "LLAMA_CPP_API_KEY", "not-needed") or "not-needed")
            except Exception:
                pass

            self.memory = Memory.from_config(config.MEM0_CONFIG)
            llm_provider = ""
            llm_model = ""
            try:
                llm_cfg = getattr(config, "MEM0_CONFIG", {}).get("llm", {}) if isinstance(getattr(config, "MEM0_CONFIG", None), dict) else {}
                llm_provider = str((llm_cfg.get("provider") or "")).strip()
                llm_model = str(((llm_cfg.get("config") or {}) or {}).get("model") or "").strip() if isinstance(llm_cfg, dict) else ""
            except Exception:
                llm_provider = ""
                llm_model = ""
            logger.info(
                f"Mem0 initialized (LLM provider: {llm_provider or 'unknown'} model: {llm_model or 'unknown'}, "
                f"Embed: {config.MEMORY_EMBEDDING_MODEL})"
            )
        except Exception as e:
            logger.error(f"Mem0 initialization failed: {e}")
            self.memory = None
            self.enabled = False

    def should_save_memory(self, text: str) -> bool:
        """
        Heuristic filter to reduce memory pollution + avoid unnecessary fact-extraction calls.
        We only attempt to save when the message is likely to contain durable personal facts.
        """
        if not text or not text.strip():
            return False

        t = text.strip()

        # Normalize whitespace for pattern checks
        compact = re.sub(r"\s+", "", t)

        # Strong signals (allow optional spaces between syllables)
        if re.search(r"생\s*일", t):
            return True
        if re.search(r"기\s*념\s*일", t):
            return True
        if re.search(r"내\s*이\s*름", t) or "이름은" in compact:
            return True
        if re.search(r"(좋아|싫어|선호|취향)", t):
            return True
        if re.search(r"(알레르기|못먹어|안먹어)", t):
            return True
        if re.search(r"(목표|계획|예정|할거야|하려고)", t):
            return True
        if re.search(r"(사는곳|살아|거주|직업|회사|학교)", t):
            return True

        return False

    def should_save_assistant_memory(self, user_text: str, assistant_text: str) -> bool:
        """
        Decide whether to save the assistant's own "profile" facts to long-term memory.

        Motivation:
        - User asks about the bot ("너가 좋아하는 과일은?")
        - Bot answers with a stable preference ("수박")
        - We want the bot to be able to recall its own stated preferences later.

        Note:
        - We intentionally keep this conservative to avoid storing hallucinated/general filler.
        """
        if not bool(getattr(config, "MEMORY_SAVE_ASSISTANT_FACTS", True)):
            return False

        u = (user_text or "").strip()
        a = (assistant_text or "").strip()
        if not u or not a:
            return False

        # 1) Only when the user is asking ABOUT the assistant (2nd-person / AI_NAME)
        compact_u = re.sub(r"\s+", "", u)
        ai_name = str(getattr(config, "AI_NAME", "") or "").strip()
        compact_ai = re.sub(r"\s+", "", ai_name)
        asked_about_ai = False
        if compact_ai and compact_ai in compact_u:
            asked_about_ai = True
        if any(k in compact_u for k in ("너", "너가", "너는", "너의", "당신", "니가", "넌")):
            asked_about_ai = True
        if not asked_about_ai:
            return False

        # 2) Only when assistant answer looks like a durable self-fact (preference / identity)
        if not re.search(r"(저는|제가|나는|내가|제\s*)", a):
            return False

        # Use the same "durable fact" keywords as user memory, but keep it narrow.
        if re.search(r"(좋아|싫어|선호|취향)", a):
            return True
        if re.search(r"(알레르기|못먹어|안먹어)", a):
            return True
        if re.search(r"생\s*일", a) or re.search(r"기\s*념\s*일", a):
            return True
        if "이름" in a or "이름은" in re.sub(r"\s+", "", a):
            return True

        return False

    def save_memory(self, user_name: str, text: str) -> Optional[Dict]:
        """
        Saves a text snippet to mem0 memory.
        Mem0 automatically extracts and stores relevant facts using Ollama LLM.
        
        Note: This runs synchronously. For async usage, call via asyncio.to_thread
        or use save_memory_async.
        """
        if not self.enabled or not self.memory:
            return None
        
        if not text or not text.strip():
            return None
            
        try:
            result = self.memory.add(
                text,
                user_id=user_name,
                metadata={"source": "voice_chat"}
            )

            # Always print completion + stored sentences (requested)
            # NOTE: mem0 may extract facts and store them as separate "memory" items.
            try:
                stored_items = []
                if isinstance(result, dict):
                    stored_items = list(result.get("results", []) or [])

                saved_texts: list[str] = []
                for item in stored_items:
                    if not isinstance(item, dict):
                        continue
                    # Try common keys used by mem0
                    m = item.get("memory") or item.get("text") or item.get("fact") or ""
                    m = str(m).strip() if m is not None else ""
                    if m:
                        saved_texts.append(m)

                # Terminal output (Korean)
                print("저장 완료")
                print(f"입력 문장: {text.strip()}")
                if saved_texts:
                    for s in saved_texts:
                        print(f"저장된 문장: {s}")
                else:
                    print("저장된 문장: (없음)")
            except Exception:
                # Fallback: still print completion
                try:
                    print("저장 완료")
                    print(f"입력 문장: {text.strip()}")
                    print("저장된 문장: (출력 실패)")
                except Exception:
                    pass

            if result and result.get("results"):
                logger.debug(f"[Memory] Saved for {user_name}: {result}")
            return result
        except Exception as e:
            logger.error(f"[Memory] Save error: {e}")
            return None

    @staticmethod
    def _extract_saved_texts(result: Optional[Dict]) -> List[str]:
        """
        Best-effort extraction of stored memory strings from mem0 add() result.
        """
        try:
            stored_items = []
            if isinstance(result, dict):
                stored_items = list(result.get("results", []) or [])

            saved_texts: list[str] = []
            for item in stored_items:
                if not isinstance(item, dict):
                    continue
                m = item.get("memory") or item.get("text") or item.get("fact") or ""
                m = str(m).strip() if m is not None else ""
                if m:
                    saved_texts.append(m)
            return saved_texts
        except Exception:
            return []

    @staticmethod
    def _print_save_result(title: str, input_text: str, result: Optional[Dict]) -> None:
        """
        Always print completion + stored sentences (terminal).
        """
        try:
            saved_texts = MemoryManager._extract_saved_texts(result)
            print("저장 완료" + (f" ({title})" if title else ""))
            print(f"입력 문장: {input_text.strip()}")
            if saved_texts:
                for s in saved_texts:
                    print(f"저장된 문장: {s}")
            else:
                print("저장된 문장: (없음)")
        except Exception:
            try:
                print("저장 완료")
                print(f"입력 문장: {input_text.strip()}")
                print("저장된 문장: (출력 실패)")
            except Exception:
                pass

    def save_turn_reflection(
        self,
        turn_id: Optional[int],
        user_name: str,
        user_text: str,
        assistant_name: str,
        assistant_text: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Reflect on a full conversation turn (user + assistant) and let the LLM decide what to store.

        - Optionally stores raw turn text (infer=False) for episodic recall
        - Stores extracted facts for the user (infer=True, prompt-guided)
        - Optionally stores extracted facts for the assistant persona (infer=True, prompt-guided)

        This is designed to run off the critical path (background worker).
        """
        if not self.enabled or not self.memory:
            return

        u = (user_text or "").strip()
        a = (assistant_text or "").strip()
        if not u and not a:
            return

        base_meta = {"source": "voice_chat", "type": "turn_reflection"}
        if isinstance(turn_id, int):
            base_meta["turn_id"] = turn_id
        if isinstance(metadata, dict) and metadata:
            try:
                base_meta.update(metadata)
            except Exception:
                pass

        # Raw episodic turn (no inference) — useful for "우리가 그때 뭐 했지?" type queries
        if bool(getattr(config, "MEMORY_STORE_RAW_TURNS", True)):
            raw_turn_text = f"{user_name}: {u}\n{assistant_name}: {a}".strip()
            try:
                r0 = self.memory.add(
                    raw_turn_text,
                    user_id=user_name,
                    metadata={**base_meta, "kind": "raw_turn"},
                    infer=False,
                )
                self._print_save_result("원문", raw_turn_text, r0 if isinstance(r0, dict) else None)
            except Exception as e:
                logger.error(f"[Memory] Raw turn save error: {e}")

        # User long-term facts (LLM decides; may return empty)
        try:
            prompt_t = str(getattr(config, "MEMORY_TURN_REFLECTION_PROMPT_USER", "") or "").strip()
            prompt = (
                prompt_t.format(
                    user_name=user_name,
                    user_text=u,
                    ai_name=assistant_name,
                    assistant_text=a,
                )
                if prompt_t
                else None
            )
            messages = [
                {"role": "user", "content": f"{user_name}: {u}"},
                {"role": "assistant", "content": f"{assistant_name}: {a}"},
            ]
            r1 = self.memory.add(
                messages,
                user_id=user_name,
                metadata={**base_meta, "kind": "user_facts"},
                infer=True,
                prompt=prompt,
            )
            input_text = f"{user_name}: {u} / {assistant_name}: {a}"
            self._print_save_result("유저", input_text, r1 if isinstance(r1, dict) else None)
        except Exception as e:
            logger.error(f"[Memory] Turn reflection(user) error: {e}")

        # Assistant persona facts (optional; LLM decides; may return empty)
        if bool(getattr(config, "MEMORY_SAVE_ASSISTANT_FACTS", True)):
            try:
                prompt_t2 = str(getattr(config, "MEMORY_TURN_REFLECTION_PROMPT_ASSISTANT", "") or "").strip()
                prompt2 = (
                    prompt_t2.format(
                        user_name=user_name,
                        user_text=u,
                        ai_name=assistant_name,
                        assistant_text=a,
                    )
                    if prompt_t2
                    else None
                )
                messages2 = [
                    {"role": "user", "content": f"{user_name}: {u}"},
                    {"role": "assistant", "content": f"{assistant_name}: {a}"},
                ]
                r2 = self.memory.add(
                    messages2,
                    user_id=assistant_name,
                    metadata={**base_meta, "kind": "assistant_facts"},
                    infer=True,
                    prompt=prompt2,
                )
                input_text2 = f"{user_name}: {u} / {assistant_name}: {a}"
                self._print_save_result("어시스턴트", input_text2, r2 if isinstance(r2, dict) else None)
            except Exception as e:
                logger.error(f"[Memory] Turn reflection(assistant) error: {e}")

    def search_memory(
        self,
        query_text: str,
        user_name: str = None,
        limit: int = 3,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Searches for relevant memories using mem0.
        Uses vector search only (rerank=False) for faster response - NO LLM call.
        """
        if not self.enabled or not self.memory:
            return []
        
        if not query_text or not query_text.strip():
            return []
            
        try:
            results = self.memory.search(
                query_text,
                user_id=user_name,
                limit=limit,
                filters=(filters if isinstance(filters, dict) else None),
                rerank=False
            )
            
            memories = []
            if results and results.get("results"):
                for item in results["results"]:
                    memories.append({
                        "text": item.get("memory", ""),
                        "user": user_name,
                        "score": item.get("score", 0)
                    })

            # Simple completion print (requested)
            top_score = 0
            try:
                top_score = max((m.get("score", 0) or 0) for m in memories) if memories else 0
            except Exception:
                top_score = 0
            print(f"[Memory] search_done user={user_name} hits={len(memories)} top_score={top_score}")

            return memories
        except Exception as e:
            logger.error(f"[Memory] Search error: {e}")
            return []

    def get_memory_context(self, query_text: str, user_name: str) -> str:
        """
        Returns a formatted string of relevant memories for the context.
        """
        # In reflect mode, prefer curated facts by default (raw turns are large/noisy).
        filters = None
        if str(getattr(config, "MEMORY_SAVE_MODE", "heuristic") or "heuristic").strip().lower() == "reflect":
            # If this is the assistant's own memory space, use assistant_facts; otherwise use user_facts.
            ai_name = str(getattr(config, "AI_NAME", "") or "").strip()
            if ai_name and str(user_name) == ai_name:
                filters = {"kind": "assistant_facts"}
            else:
                filters = {"kind": "user_facts"}

        memories = self.search_memory(query_text, user_name, filters=filters)
        if not memories:
            return ""
            
        context = "\nRelevant Long-term Memories:\n"
        for mem in memories:
            context += f"- {mem['text']}\n"
            
        return context

    def get_all_memories(self, user_name: str) -> List[Dict]:
        """
        Retrieves all memories for a specific user.
        """
        if not self.enabled or not self.memory:
            return []
            
        try:
            result = self.memory.get_all(user_id=user_name)
            return result.get("results", []) if result else []
        except Exception as e:
            logger.error(f"[Memory] Get all error: {e}")
            return []

    def delete_all_memories(self, user_name: str) -> bool:
        """
        Deletes all memories for a specific user.
        """
        if not self.enabled or not self.memory:
            return False
            
        try:
            self.memory.delete_all(user_id=user_name)
            logger.info(f"[Memory] Deleted all memories for {user_name}")
            return True
        except Exception as e:
            logger.error(f"[Memory] Delete error: {e}")
            return False
