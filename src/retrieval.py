import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CHROMA_DIR, TOP_K, SIMILARITY_THRESHOLD, EMBEDDING_MODEL

import chromadb
from chromadb.utils import embedding_functions
from src.reranker import Reranker  # <-- Import reranker

class MedicalRetriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # ? Using the NEW semantic collection
        self.collection = self.client.get_collection("medical_knowledge_md")
        self.threshold = SIMILARITY_THRESHOLD
        
        # Load reranker
        self.reranker = Reranker()
        
        print(f"? Retriever ready. Collection: medical_knowledge_md (Threshold={self.threshold})")

    def search(self, query: str) -> list:
        """
        1. ChromaDB se top 20 chunks lao (High Recall)
        2. Reranker se unko rank karo
        3. Sirf top 3 return karo (High Precision)
        """
        # Step 1: Fetch 20 chunks from ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=20  # <-- 20 kyun? Upar reason parha hai
        )
        
        if not results or not results['documents']:
            return []
        
        # Format documents for reranker
        docs = []
        for i, chunk in enumerate(results['documents'][0]):
            similarity = round(1 - results['distances'][0][i], 3)
            # Only pass if above threshold
            if similarity >= self.threshold:
                docs.append({
                    "text": chunk,
                    "source": results['metadatas'][0][i].get("source", "unknown"),
                    "similarity_score": similarity
                })
        
        if not docs:
            return []
        
        # Step 2: Rerank the 20 chunks
        reranked = self.reranker.rerank(query, docs, top_k=3)  # <-- Top 3 kyun? Upar reason parha hai
        
        return reranked

    def search_with_context(self, query: str) -> str:
        results = self.search(query)
        if not results:
            return ""
        
        context_text = ""
        for item in results:
            context_text += f"Source: {item['source']} (Rerank Score: {item['rerank_score']:.4f})\n{item['text']}\n\n"
        
        return context_text.strip()
