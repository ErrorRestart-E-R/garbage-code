from mem0 import Memory
import config
from logger import setup_logger

logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)


class MemoryManager:
    def __init__(self):
        """
        Initialize mem0 Memory with llama.cpp + ChromaDB configuration.
        """
        self.memory = Memory.from_config(config.MEM0_CONFIG)
        logger.info("Mem0 Memory initialized with llama.cpp + ChromaDB")

    def save_memory(self, user_name: str, text: str):
        """
        Saves a text snippet to mem0 memory.
        mem0 automatically extracts and stores relevant facts.
        """
        try:
            result = self.memory.add(
                text,
                user_id=user_name,
                metadata={"source": "voice_chat"}
            )
            if result and result.get("results"):
                logger.debug(f"[Memory] Saved for {user_name}: {result}")
            return result
        except Exception as e:
            logger.error(f"[Memory] Save error: {e}")
            return None

    def search_memory(self, query_text: str, user_name: str = None, limit: int = 3):
        """
        Searches for relevant memories using mem0.
        """
        try:
            results = self.memory.search(
                query_text,
                user_id=user_name,
                limit=limit
            )
            
            memories = []
            if results and results.get("results"):
                for item in results["results"]:
                    memories.append({
                        "text": item.get("memory", ""),
                        "user": user_name,
                        "score": item.get("score", 0)
                    })
            
            return memories
        except Exception as e:
            logger.error(f"[Memory] Search error: {e}")
            return []

    def get_memory_context(self, query_text: str, user_name: str):
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

    def get_all_memories(self, user_name: str):
        """
        Retrieves all memories for a specific user.
        """
        try:
            result = self.memory.get_all(user_id=user_name)
            return result.get("results", []) if result else []
        except Exception as e:
            logger.error(f"[Memory] Get all error: {e}")
            return []
