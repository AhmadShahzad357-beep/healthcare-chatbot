import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CHROMA_DIR

import chromadb

def verify_accuracy():
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    try:
        collection = chroma_client.get_collection("medical_knowledge_md")
    except:
        print("❌ 'medical_knowledge_md' collection not found. Run ingest_md_to_chroma.py first.")
        return
    
    # 1. Total chunks count
    total_chunks = collection.count()
    print(f"\n📊 Total chunks in ChromaDB (MD collection): {total_chunks}")
    
    # 2. Sample chunks print (Manual Verification)
    print("\n🔍 Sample Chunks (Check if they look like proper Markdown content):")
    sample_chunks = collection.get(limit=3)
    if sample_chunks and sample_chunks['documents']:
        for i, doc in enumerate(sample_chunks['documents'], 1):
            print(f"\n--- Sample {i} ---")
            preview = doc[:400].replace('\n', ' ')
            print(preview + "..." if len(doc) > 400 else doc)
    
    # 3. Word count comparison
    md_folder = Path("data/md_from_pdfs")
    total_md_words = 0
    if md_folder.exists():
        for md_file in md_folder.glob("*.md"):
            text = md_file.read_text(encoding="utf-8")
            total_md_words += len(text.split())
    
    total_chroma_words = 0
    batch_size = 100
    offset = 0
    while True:
        batch = collection.get(limit=batch_size, offset=offset)
        if not batch or not batch['documents']:
            break
        for doc in batch['documents']:
            total_chroma_words += len(doc.split())
        offset += batch_size
        if len(batch['documents']) < batch_size:
            break
    
    print(f"\n📝 Total Words in Original MD files: {total_md_words}")
    print(f"📝 Total Words in ChromaDB chunks: {total_chroma_words}")
    
    diff = abs(total_md_words - total_chroma_words)
    if diff < 100:
        print("✅ Word count matches! Data is 99% intact.")
    else:
        print(f"⚠️ Word count mismatch (Difference: {diff}). Check chunking logic.")

if __name__ == "__main__":
    verify_accuracy()
