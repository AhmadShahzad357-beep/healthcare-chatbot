import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.retrieval import MedicalRetriever
import requests
import os
from dotenv import load_dotenv
load_dotenv()

def check_faithfulness(query: str, answer: str, context: str) -> bool:
    """Check karein ke answer context se bana hai ya nahi (Simple heuristic)"""
    # Agar answer mein context ke words nahi hain, toh hallucination
    context_words = set(context.lower().split())
    answer_words = set(answer.lower().split())
    overlap = context_words.intersection(answer_words)
    
    # Agar 30% se kam overlap hai, toh hallucination
    if len(overlap) / len(answer_words) < 0.3:
        return False
    return True

def test_faithfulness():
    print("="*50)
    print("🧪 Faithfulness Test (Hallucination Check)")
    print("="*50)
    
    retriever = MedicalRetriever()
    
    test_queries = [
        "What are the symptoms of diabetes?",
        "How to treat malaria?",
        "What is the normal blood pressure range?"
    ]
    
    for q in test_queries:
        chunks = retriever.search(q)
        if not chunks:
            print(f"❌ No context for: {q}")
            continue
        
        context = "\n".join([c['text'] for c in chunks])
        
        # LLM se jawab lenay ki simulation (hum direct response nahi le sakte, is liye placeholder)
        # Main ne yahan assumption rakhi hai ke LLM ne context se jawab banaya hai.
        # Asli test mein aap Groq call karke response lein aur check karein.
        print(f"✅ Context length for '{q[:30]}...': {len(context)} chars")
    
    print("✅ Faithfulness test complete. (Placeholder)")

if __name__ == "__main__":
    test_faithfulness()
