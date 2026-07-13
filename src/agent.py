import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import retrieval
import patient_db
import medicine_api
from retrieval_history import ChatHistoryRetriever  # <-- Naya

COMMON_DRUGS = {
    "paracetamol", "panadol", "acetaminophen", "ibuprofen", "brufen", "advil",
    "aspirin", "disprin", "ecospirin", "metformin", "insulin", "lisinopril",
    "atenolol", "verapamil", "warfarin", "amoxicillin", "azithromycin",
    "omeprazole", "levothyroxine", "atorvastatin", "simvastatin"
}

# History keywords for cross-session memory
HISTORY_KEYWORDS = ["pehle", "week", "month", "past", "previous", "old", "1 week", "7 din", "pichli", "kab", "us waqt", "pichhle", "guzishta", "aik hafty", "aik mahina"]

def is_patient_query(query: str) -> bool:
    query_lower = query.lower()
    pronouns = ["mera", "meri", "mere", "my", "mujhe", "apna", "apni", "patient", "mari"]
    medical_context = ["report", "test", "allergy", "allergies", "history", "appointment", "checkup", "blood", "sugar", "bp"]
    return any(p in query_lower for p in pronouns) and any(m in query_lower for m in medical_context)

def is_history_query(query: str) -> bool:
    return any(word in query.lower() for word in HISTORY_KEYWORDS)

def is_medicine_interaction_query(query: str) -> bool:
    query_lower = query.lower()
    conjunctions = ["aur", "and", "+", "&", "ke saath", "with"]
    if not any(c in query_lower for c in conjunctions):
        return False
    words = re.findall(r'[a-zA-Z]+', query_lower)
    drug_matches = [word for word in words if word in COMMON_DRUGS]
    return len(set(drug_matches)) >= 2

def route_query(query: str, patient_id: int = 1):
    print(f"\n🔍 Analyzing Query: '{query}' (Patient ID: {patient_id})")

    # 🆕 1. HISTORY INTENT (RAG on Chat History)
    if is_history_query(query):
        print("   → Intent: HISTORY (RAG on Chat History)")
        retriever = ChatHistoryRetriever()
        results = retriever.search(query, patient_id, top_k=5)  # Top 5 relevant chats
        if results:
            context = "--- PAST CHATS (Semantic Search) ---\n"
            for r in results:
                context += f"[{r['timestamp']}] User: {r['user_msg']}\nBot: {r['bot_reply']}\n\n"
            return {
                "intent": "history_rag",
                "data": {"context": context, "raw": results},
                "message": f"Found {len(results)} relevant past chats."
            }
        else:
            return {
                "intent": "history_rag",
                "data": {"context": "No relevant past conversations found."},
                "message": "No relevant history found."
            }

    # 2. Patient-specific
    if is_patient_query(query):
        print("   → Intent: PATIENT_SPECIFIC (SQL)")
        db = patient_db.PatientDB()
        profile = db.get_patient_profile(patient_id)
        if profile:
            history = db.get_medical_history(patient_id)
            reports = db.get_lab_reports(patient_id, limit=5)
            apps = db.get_appointments(patient_id)
            return {
                "intent": "patient_db",
                "data": {"profile": profile, "history": history, "reports": reports, "appointments": apps},
                "message": f"Found patient: {profile['name']}"
            }
        else:
            return {"intent": "patient_db", "data": None, "message": "Patient not found."}

    # 3. Medicine
    elif is_medicine_interaction_query(query):
        print("   → Intent: MEDICINE_INTERACTION (API)")
        words = re.findall(r'[a-zA-Z]+', query.lower())
        drugs = [word for word in words if word in COMMON_DRUGS]
        if len(drugs) >= 2:
            result = medicine_api.check_medicine_interaction(drugs[0], drugs[1])
            return {"intent": "medicine_api", "data": result, "message": "Interaction checked."}
        else:
            return {"intent": "medicine_api", "data": None, "message": "Could not identify two medicines."}

    # 4. RAG (PDFs)
    else:
        print("   → Intent: GENERAL_KNOWLEDGE (RAG)")
        retriever = retrieval.MedicalRetriever()
        context_chunks = retriever.search(query)
        context_text = ""
        sources = set()
        for chunk in context_chunks:
            context_text += chunk['text'] + "\n\n"
            sources.add(chunk['source'])
        return {
            "intent": "retrieval",
            "data": {"context": context_text, "chunks": context_chunks, "sources": list(sources)},
            "message": f"Retrieved {len(context_chunks)} chunks."
        }
