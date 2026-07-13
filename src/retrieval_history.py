import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CHROMA_DIR

import chromadb

class ChatHistoryRetriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            self.collection = self.client.get_collection("chat_history")
            print("✅ ChatHistoryRetriever ready.")
        except:
            print("⚠️ Chat history collection not found. Run ingest_chat_history.py first.")
            self.collection = None

    def search(self, query: str, patient_id: int, top_k: int = 3) -> list:
        """Patient ki purani chats mein search karein"""
        if not self.collection:
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            # Sirf us patient ki chats search karein
            where={"patient_id": patient_id}
        )
        
        if not results or not results['documents']:
            return []
        
        docs = []
        for i, doc in enumerate(results['documents'][0]):
            docs.append({
                "text": doc,
                "timestamp": results['metadatas'][0][i].get("timestamp", ""),
                "session": results['metadatas'][0][i].get("session_id", ""),
                "user_msg": results['metadatas'][0][i].get("user_msg", ""),
                "bot_reply": results['metadatas'][0][i].get("bot_reply", "")
            })
        return docs
