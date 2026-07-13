import sys
import sqlite3
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CHROMA_DIR, EMBEDDING_MODEL, DB_PATH

import chromadb
from chromadb.utils import embedding_functions

def main():
    print("="*50)
    print("📂 Ingesting Chat History into ChromaDB")
    print("="*50)
    
    # 1. SQLite se chats fetch karein
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT patient_id, user_msg, bot_reply, timestamp, session_id 
        FROM conversation_logs 
        ORDER BY timestamp DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("❌ No chat history found.")
        return
    print(f"✅ Found {len(rows)} chat entries.")
    
    # 2. ChromaDB setup
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        chroma_client.delete_collection("chat_history")
        print("   Old chat_history collection deleted.")
    except:
        pass
    
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = chroma_client.create_collection(
        name="chat_history",
        embedding_function=emb_fn
    )
    print("✅ ChromaDB collection 'chat_history' ready.")
    
    # 3. Chunks banayein (Har chat ek document)
    all_docs = []
    all_metadata = []
    all_ids = []
    
    for idx, row in enumerate(rows):
        patient_id, user_msg, bot_reply, timestamp, session_id = row
        # Document format: "Patient X: User asked... Bot replied..."
        doc_text = f"Patient {patient_id} at {timestamp}:\nUser: {user_msg}\nBot: {bot_reply}"
        
        doc_id = f"chat_{idx}"
        all_docs.append(doc_text)
        all_metadata.append({
            "patient_id": patient_id,
            "timestamp": timestamp,
            "session_id": session_id,
            "user_msg": user_msg[:50],  # preview
            "bot_reply": bot_reply[:50]
        })
        all_ids.append(doc_id)
    
    # 4. ChromaDB mein add karein
    print(f"🔄 Adding {len(all_docs)} chat documents to ChromaDB...")
    batch_size = 100
    for i in range(0, len(all_docs), batch_size):
        collection.add(
            documents=all_docs[i:i+batch_size],
            metadatas=all_metadata[i:i+batch_size],
            ids=all_ids[i:i+batch_size]
        )
    
    print(f"✅ Successfully ingested {len(all_docs)} chat documents into 'chat_history'!")
    print("📍 Collection: chat_history")
    print("="*50)

if __name__ == "__main__":
    main()
