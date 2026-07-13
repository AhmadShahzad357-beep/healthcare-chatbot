import os
import re
import sys
from pathlib import Path

# Config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PDFS_DIR, CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# ================== 1. PDF Cleaning Function ==================
def clean_pdf_text(text: str) -> str:
    """
    PDF ke raw text ko clean karein:
    - Extra line breaks hatao (hyphenation fix)
    - Headers/Footers hatao (page numbers, etc.)
    - Extra whitespace trim karo
    """
    # Line breaks ko join karein (hyphenated words ko fix)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)  # diabe-tes -> diabetes
    text = re.sub(r'\n+', ' ', text)  # Multiple newlines ko single space mein badlo
    
    # Page numbers aur common headers/footers hatao
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+/\d+', '', text)  # Date patterns remove (optional)
    
    # Extra spaces trim
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ================== 2. Chunking Function ==================
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Text ko overlapping chunks mein todna"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# ================== 3. Main Ingestion Pipeline ==================
def ingest_pdfs():
    print("="*50)
    print("📂 PDF Ingestion Pipeline Started")
    print("="*50)
    
    # 1. Check if PDFs folder exists
    if not PDFS_DIR.exists():
        print(f"❌ Folder not found: {PDFS_DIR}")
        print("   Please create 'data/pdfs/' and add some PDF files.")
        return
    
    # 2. Get all PDF files
    pdf_files = list(PDFS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {PDFS_DIR}")
        return
    print(f"✅ Found {len(pdf_files)} PDF files.")
    
    # 3. Load Embedding Model
    print(f"🔄 Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Model loaded.")
    
    # 4. Initialize ChromaDB (Persistent)
    print(f"🔄 Initializing ChromaDB at: {CHROMA_DIR}...")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Delete existing collection if any (to avoid duplicates)
    try:
        chroma_client.delete_collection("medical_knowledge")
        print("   Old collection deleted.")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name="medical_knowledge",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    )
    print("✅ ChromaDB collection ready.")
    
    # 5. Process each PDF
    all_chunks = []
    all_metadata = []
    all_ids = []
    
    for idx, pdf_path in enumerate(pdf_files):
        print(f"\n📄 Processing: {pdf_path.name}")
        
        # Open PDF
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            text = clean_pdf_text(text)
            full_text += text + " "
        
        doc.close()
        
        # Chunking
        chunks = chunk_text(full_text)
        print(f"   → Generated {len(chunks)} chunks.")
        
        # Store chunks
        for i, chunk in enumerate(chunks):
            chunk_id = f"{pdf_path.stem}_{i}"
            all_chunks.append(chunk)
            all_metadata.append({
                "source": pdf_path.name,
                "chunk_index": i
            })
            all_ids.append(chunk_id)
    
    # 6. Add chunks to ChromaDB (in batches)
    print(f"\n🔄 Adding {len(all_chunks)} total chunks to ChromaDB...")
    
    # Batch size to avoid memory issues
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_ids = all_ids[i:i+batch_size]
        batch_metadata = all_metadata[i:i+batch_size]
        
        collection.add(
            documents=batch_chunks,
            metadatas=batch_metadata,
            ids=batch_ids
        )
    
    print(f"✅ Successfully ingested {len(all_chunks)} chunks into ChromaDB!")
    print(f"📍 Database saved at: {CHROMA_DIR}")
    print("="*50)

if __name__ == "__main__":
    ingest_pdfs()
