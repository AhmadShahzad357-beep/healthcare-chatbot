import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        print(f"🔄 Loading Reranker: {model_name}...")
        self.model = CrossEncoder(model_name, max_length=512)
        print("✅ Reranker loaded.")

    def rerank(self, query: str, documents: list, top_k: int = 3) -> list:
        """
        documents: list of dicts with 'text' and 'metadata'
        Returns: top_k documents sorted by relevance
        """
        if not documents:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc['text']] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Combine scores with documents
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[i])
        
        # Sort by score (highest first)
        sorted_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        return sorted_docs[:top_k]
