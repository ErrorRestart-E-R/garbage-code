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
            self.memory = Memory.from_config(config.MEM0_CONFIG)
            logger.info(f"Mem0 initialized with Ollama (LLM: {config.MEMORY_LLM_MODEL}, Embed: {config.MEMORY_EMBEDDING_MODEL})")
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

    def search_memory(self, query_text: str, user_name: str = None, limit: int = 3) -> List[Dict]:
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
        memories = self.search_memory(query_text, user_name)
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
