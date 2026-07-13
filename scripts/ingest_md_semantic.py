import sys
import re
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CHROMA_DIR, EMBEDDING_MODEL

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    print("="*50)
    print("📂 Semantic Chunking Pipeline Started")
    print("="*50)
    
    md_folder = Path("data/md_from_pdfs")
    if not md_folder.exists():
        print("❌ MD folder not found!")
        return
    
    md_files = list(md_folder.glob("*.md"))
    if not md_files:
        print("❌ No MD files found.")
        return
    print(f"✅ Found {len(md_files)} MD files.")
    
    # 1. Load model
    print(f"🔄 Loading embedding model: {EMBEDDING_MODEL}...")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    
    # 2. ChromaDB setup (Nayi collection)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        chroma_client.delete_collection("medical_knowledge_semantic")
        print("   Old semantic collection deleted.")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name="medical_knowledge_semantic",
        embedding_function=emb_fn
    )
    print("✅ ChromaDB collection 'medical_knowledge_semantic' ready.")
    
    # 3. LangChain Semantic Splitter (Paragraph, Sentence, Word fallback)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,          # Target size
        chunk_overlap=50,        # Overlap
        separators=["\n\n", "\n", ". ", " ", ""],  # Priority: paragraphs first, then sentences
        length_function=len,
    )
    
    all_chunks = []
    all_metadata = []
    all_ids = []
    
    for md_path in md_files:
        print(f"📄 Processing: {md_path.name}")
        text = md_path.read_text(encoding="utf-8")
        
        # LangChain chunking
        chunks = splitter.split_text(text)
        print(f"   → Generated {len(chunks)} semantic chunks.")
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{md_path.stem}_{i}"
            all_chunks.append(chunk)
            all_metadata.append({"source": md_path.name, "chunk_index": i, "type": "semantic"})
            all_ids.append(chunk_id)
    
    # 4. Add to ChromaDB
    print(f"\n🔄 Adding {len(all_chunks)} semantic chunks to ChromaDB...")
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            documents=all_chunks[i:i+batch_size],
            metadatas=all_metadata[i:i+batch_size],
            ids=all_ids[i:i+batch_size]
        )
    
    print(f"✅ Successfully ingested {len(all_chunks)} semantic chunks!")
    print("📍 Collection: medical_knowledge_semantic")
    print("="*50)

if __name__ == "__main__":
    main()
