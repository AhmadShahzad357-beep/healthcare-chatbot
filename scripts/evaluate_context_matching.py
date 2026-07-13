import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import our retriever (MD collection wala)
from src.retrieval import MedicalRetriever

# ========================================
# 15 Test Queries with Expected Keywords
# ========================================
TEST_QUERIES = [
    # Diabetes
    {"query": "What are the symptoms of diabetes?", "keywords": ["symptom", "thirst", "urination", "weight"]},
    {"query": "What causes high blood sugar?", "keywords": ["glucose", "insulin", "pancreas"]},
    
    # Malaria
    {"query": "How is malaria treated?", "keywords": ["artemisinin", "ACT", "combination", "therapy"]},
    {"query": "What are the signs of malaria?", "keywords": ["fever", "chills", "parasite", "plasmodium"]},
    
    # Heart Attack
    {"query": "What are the warning signs of a heart attack?", "keywords": ["chest", "pain", "discomfort", "arm", "jaw"]},
    {"query": "What causes a myocardial infarction?", "keywords": ["blockage", "artery", "plaque", "blood clot"]},
    
    # Asthma
    {"query": "What triggers an asthma attack?", "keywords": ["allergen", "smoke", "dust", "pollen"]},
    {"query": "How to treat asthma symptoms?", "keywords": ["inhaler", "bronchodilator", "steroid"]},
    
    # Hypertension (from 1384062_EN.pdf)
    {"query": "What is the normal blood pressure range?", "keywords": ["systolic", "diastolic", "120", "80", "mmhg"]},
    {"query": "What are the complications of hypertension?", "keywords": ["stroke", "kidney", "heart", "damage"]},
    
    # General / Mixed
    {"query": "What is the relationship between diabetes and heart disease?", "keywords": ["cardiovascular", "risk", "glucose"]},
    {"query": "How to prevent malaria?", "keywords": ["mosquito", "net", "repellent", "prophylaxis"]},
    {"query": "What are the side effects of high cholesterol?", "keywords": ["plaque", "artery", "stroke"]},
    {"query": "Define chronic disease.", "keywords": ["long-term", "persistent", "condition"]},
    {"query": "What is the role of insulin in the body?", "keywords": ["glucose", "blood sugar", "pancreas"]},
]

# ========================================
# Evaluation Logic
# ========================================
def evaluate_retrieval():
    print("="*60)
    print("🧪 Context Matching Evaluation (MRR & Hit Rate)")
    print("="*60)
    print(f"📊 Total Test Queries: {len(TEST_QUERIES)}")
    print("-"*60)
    
    # Initialize retriever
    retriever = MedicalRetriever()  # Ye ab medical_knowledge_md collection use karega
    
    reciprocal_ranks = []
    hit_at_1 = 0
    hit_at_3 = 0
    
    for idx, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        expected_keywords = test["keywords"]
        
        # Retrieve chunks
        results = retriever.search(query)
        
        # Check ranks
        rank_found = None
        for rank, chunk_data in enumerate(results, 1):
            chunk_text = chunk_data["text"].lower()
            # Check if ANY expected keyword exists in this chunk
            if any(keyword.lower() in chunk_text for keyword in expected_keywords):
                rank_found = rank
                break
        
        # Calculate metrics for this query
        reciprocal = 1 / rank_found if rank_found else 0
        reciprocal_ranks.append(reciprocal)
        
        if rank_found == 1:
            hit_at_1 += 1
        if rank_found and rank_found <= 3:
            hit_at_3 += 1
        
        # Print progress
        status = "✅" if rank_found else "❌"
        rank_str = f"Rank: {rank_found}" if rank_found else "Not Found"
        print(f"{idx}. {query[:50]}... \t {status} {rank_str} (Reciprocal: {reciprocal:.2f})")
    
    # ========================================
    # Final Score Calculation
    # ========================================
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    hit_rate_1 = hit_at_1 / len(TEST_QUERIES)
    hit_rate_3 = hit_at_3 / len(TEST_QUERIES)
    
    print("\n" + "="*60)
    print("📊 FINAL EVALUATION REPORT")
    print("="*60)
    print(f"🎯 MRR (Mean Reciprocal Rank): {mrr:.4f}")
    print(f"🎯 Hit Rate @1: {hit_rate_1:.2%} ({hit_at_1}/{len(TEST_QUERIES)})")
    print(f"🎯 Hit Rate @3: {hit_rate_3:.2%} ({hit_at_3}/{len(TEST_QUERIES)})")
    print("-"*60)
    
    # The "0.8" Threshold Check (Which we discussed earlier)
    if mrr >= 0.8:
        print("✅ PASSED! MRR is >= 0.8. Retrieval quality is EXCELLENT.")
        print("   System is ready for production.")
    elif mrr >= 0.6:
        print("⚠️ MRR is between 0.6 and 0.8. Consider adjusting chunk size or threshold.")
    else:
        print("❌ MRR is below 0.6. Retrieval quality is POOR.")
        print("   Try: 1. Reducing chunk size to 300. 2. Changing embedding model.")
    
    print("="*60)

if __name__ == "__main__":
    evaluate_retrieval()
