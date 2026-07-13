import sys
import re
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

import chromadb
from chromadb.utils import embedding_functions

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def main():
    # 1. MD files read karein (sirf PDFs wali folder)
    md_folder = Path("data/md_from_pdfs")
    all_md_files = list(md_folder.glob("*.md")) if md_folder.exists() else []
    
    if not all_md_files:
        print("❌ No MD files found in data/md_from_pdfs/")
        return
    
    print(f"✅ Found {len(all_md_files)} MD files.")
    
    # 2. ChromaDB setup (Nayi collection - MD wali)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    try:
        chroma_client.delete_collection("medical_knowledge_md")
        print("   Old 'medical_knowledge_md' deleted.")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name="medical_knowledge_md",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    )
    print("✅ ChromaDB collection 'medical_knowledge_md' ready.")
    
    # 3. Process MD files
    all_chunks = []
    all_metadata = []
    all_ids = []
    
    for md_path in all_md_files:
        print(f"📄 Processing: {md_path.name}")
        text = md_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        print(f"   → Generated {len(chunks)} chunks.")
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{md_path.stem}_{i}"
            all_chunks.append(chunk)
            all_metadata.append({
                "source": md_path.name,
                "type": "md_converted",
                "chunk_index": i
            })
            all_ids.append(chunk_id)
    
    # 4. ChromaDB mein add karein
    print(f"\n🔄 Adding {len(all_chunks)} chunks to ChromaDB...")
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            documents=all_chunks[i:i+batch_size],
            metadatas=all_metadata[i:i+batch_size],
            ids=all_ids[i:i+batch_size]
        )
    
    print(f"✅ Successfully ingested {len(all_chunks)} chunks into 'medical_knowledge_md'!")
    print(f"📍 Database saved at: {CHROMA_DIR}")

if __name__ == "__main__":
    main()
