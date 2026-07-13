import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
import uvicorn
from dotenv import load_dotenv
from twilio.twiml.messaging_response import MessagingResponse

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

sys.path.append(str(Path(__file__).resolve().parent))
from config import GROQ_MODEL, TEMPERATURE, MAX_TOKENS
import src.agent as agent
from src.patient_db import PatientDB
from src.safety_guard import SafetyGuard

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"🔑 API Key Loaded: {'Yes' if GROQ_API_KEY else 'No'}")

try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception as e:
    print(f"❌ Groq init error: {e}")
    groq_client = None

db = PatientDB()

BASE_SYSTEM_PROMPT = """
You are a strict medical assistant.
🔒 RULES:
1. ONLY answer based on "MEDICAL CONTEXT". If not present, say "I don't know".
2. NEVER diagnose. Always recommend: "Please consult a doctor."
"""

def process_query(query: str, patient_id: int) -> str:
    routed = agent.route_query(query, patient_id=patient_id)
    intent = routed['intent']
    context_str = ""
    
    if intent == "patient_db":
        data = routed['data']
        if data and data.get('profile'):
            profile = data['profile']
            context_str = f"Patient: {profile['name']}, Age: {profile['age']}, Blood Group: {profile['blood_group']}, Allergies: {profile['allergies']}\n"
    elif intent == "medicine_api":
        data = routed['data']
        if data and data.get('interaction_found'):
            context_str = f"Interaction: {data['severity']}. {data['message']}"
        else:
            context_str = "No major interaction."
    elif intent == "retrieval":
        context_str = routed['data'].get('context', 'No relevant medical data.')
    else:
        context_str = "No data available."
    
    system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n--- CONTEXT ---\n{context_str}\n--- END ---"
    
    if "No relevant" in system_prompt or "No data" in system_prompt:
        return "I don't know. Please consult a doctor."
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# ========================================
# WHATSAPP WEBHOOK (Full System)
# ========================================
@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    try:
        form = await request.form()
        incoming_msg = form.get('Body', '').strip()
        sender = form.get('From', '').replace("whatsapp:", "")
        
        print(f"📩 WhatsApp: {sender} -> {incoming_msg}")
        
        reply = process_query(incoming_msg, patient_id=1)
        
        resp = MessagingResponse()
        resp.message(reply)
        return PlainTextResponse(content=str(resp), media_type="application/xml")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return PlainTextResponse(content="Error", status_code=500)

# ========================================
# CHAT ENDPOINT (Web)
# ========================================
class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    reply = process_query(req.query, patient_id=1)
    return {"response": reply}

@app.get("/patients")
async def get_patients():
    patients = db.get_all_patients()
    return {"patients": patients}

frontend_path = Path(__file__).resolve().parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
