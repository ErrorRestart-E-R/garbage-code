import chromadb
from chromadb.utils import embedding_functions
import datetime
import os

class MemoryManager:
    def __init__(self, db_path="./memory_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Use a lightweight embedding model
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="neuro_memory",
            embedding_function=self.embedding_fn
        )

    def save_memory(self, user_name, text):
        """
        Saves a text snippet to the vector DB with metadata.
        """
        timestamp = datetime.datetime.now().isoformat()
        
        # Generate a unique ID (simple timestamp + hash based)
        doc_id = f"{user_name}_{timestamp}"
        
        self.collection.add(
            documents=[text],
            metadatas=[{"user": user_name, "timestamp": timestamp}],
            ids=[doc_id]
        )
        print(f"[Memory] Saved: {text} (User: {user_name})")

    def search_memory(self, query_text, user_name=None, limit=3):
        """
        Searches for relevant memories.
        If user_name is provided, filters by that user.
        """
        where_filter = {"user": user_name} if user_name else None
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=limit,
            where=where_filter
        )
        
        # Parse results
        memories = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                memories.append({
                    "text": doc,
                    "user": metadata['user'],
                    "timestamp": metadata['timestamp']
                })
        
        return memories

    def get_memory_context(self, query_text, user_name):
        """
        Returns a formatted string of relevant memories for the context.
        """
        memories = self.search_memory(query_text, user_name)
        if not memories:
            return ""
            
        context = "\nRelevant Long-term Memories:\n"
        for mem in memories:
            context += f"- {mem['text']} (from {mem['user']})\n"
            
        return context
