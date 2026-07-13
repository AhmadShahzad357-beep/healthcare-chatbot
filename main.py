import os
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

sys.path.append(str(Path(__file__).resolve().parent))
from config import GROQ_MODEL, TEMPERATURE, MAX_TOKENS

import src.agent as agent
from src.safety_guard import SafetyGuard

# Groq API
try:
    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except ImportError:
    print("⚠️ Groq library not installed. Run: pip install groq")
    groq_client = None
except Exception as e:
    print(f"⚠️ Groq client error: {e}")
    groq_client = None

# ================== Safety Guard Instance ==================
safety = SafetyGuard()

# ================== System Prompt ==================
SYSTEM_PROMPT = """
You are a helpful medical information assistant for a hospital. 
IMPORTANT RULES:
1. ONLY use the provided "Context" to answer the user's question. 
2. If the context does NOT contain the answer, clearly say: "Mujhe nahi pata. Kripya doctor se rabta karein."
3. NEVER diagnose, prescribe medication, or give treatment advice. Always recommend consulting a doctor.
4. Always mention the sources (e.g., "Source: WHO Diabetes.pdf").
5. Keep the answer simple, clear, and in the same language as the user's question (Urdu or English).
6. If patient data (allergies, reports) is shown, summarize it factually without making medical judgments.
"""

# ================== Build Prompt based on Intent ==================
def build_prompt(query: str, routed_data: dict) -> str:
    """Agent se aaye hue data ko LLM-friendly prompt mein convert karein"""
    
    intent = routed_data['intent']
    user_message = f"User Question: {query}\n\n"
    
    if intent == "patient_db":
        data = routed_data['data']
        if data and data.get('profile'):
            profile = data['profile']
            history = data.get('history', [])
            reports = data.get('reports', [])
            apps = data.get('appointments', [])
            
            context = f"Patient Profile:\n- Name: {profile['name']}\n- Age: {profile['age']}\n- Allergies: {profile['allergies']}\n"
            context += f"Blood Group: {profile['blood_group']}\n\n"
            
            if history:
                context += "Medical History:\n"
                for h in history[:5]:
                    context += f"- {h['condition']} (Diagnosed: {h['diagnosed_date']}, Status: {h['status']})\n"
                context += "\n"
            
            if reports:
                context += "Recent Lab Reports:\n"
                for r in reports[:5]:
                    context += f"- {r['test_name']}: {r['result']} {r['unit']} (Date: {r['date']})\n"
                context += "\n"
            
            if apps:
                context += "Appointments:\n"
                for a in apps[:3]:
                    context += f"- {a['date']} at {a['time']} with {a['doctor']} ({a['status']})\n"
                context += "\n"
            
            return user_message + "Context (Patient Data):\n" + context
    
    elif intent == "medicine_api":
        data = routed_data['data']
        if data:
            if data.get('interaction_found'):
                context = f"Medicine Interaction Found:\n- Severity: {data['severity']}\n- Details: {data['message']}\n"
            else:
                context = f"No major interaction found.\n{data.get('message', '')}\n"
            return user_message + "Context (Medicine API):\n" + context
    
    elif intent == "retrieval":
        data = routed_data['data']
        context = data.get('context', '')
        sources = data.get('sources', [])
        if context:
            return user_message + f"Context (Retrieved from {len(sources)} documents):\n{context}\n\nSources: {', '.join(sources)}"
        else:
            return user_message + "Context: No relevant information found in medical documents.\n"
    
    return user_message + "Context: No data available.\n"

# ================== Main Query Handler ==================
def process_query(query: str) -> str:
    """Query process karein: Route -> Build Prompt -> LLM -> Return Answer"""
    
    if not groq_client:
        return "⚠️ Groq client not initialized. Please check your API key and install groq (pip install groq)."
    
    # 1. Route the query
    routed = agent.route_query(query)
    
    # 2. Build the prompt
    user_content = build_prompt(query, routed)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_content}\n\nNow answer the user's question based ONLY on the context above."
    
    # 3. Safety Check (Prevention: agar context empty hai toh LLM ko bhejna hi band karein)
    if "No relevant information found" in user_content or "No data available" in user_content:
        return "❌ I couldn't find relevant information in my medical database for your question. Please consult a doctor for accurate advice."
    
    # 4. Call LLM (Groq)
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        raw_answer = response.choices[0].message.content
        
        # 5. Safety Guard (Final Hallucination Check)
        if safety.check_hallucination(raw_answer, user_content):
            # Agar safe hai toh return karein
            return raw_answer
        else:
            # Agar hallucination detect ho toh fallback
            return "⚠️ I noticed my previous response might not be fully accurate. Let me play it safe: Please consult a doctor for this specific medical question."
    
    except Exception as e:
        return f"❌ Error calling LLM: {str(e)}"

# ================== Testing ==================
if __name__ == "__main__":
    print("="*50)
    print("🧠 Healthcare RAG Chatbot - Main Orchestrator Test")
    print("="*50)
    print("⚠️ Make sure you have set GROQ_API_KEY in .env file.")
    print("   Get your free key from: https://console.groq.com/keys")
    print("="*50)
    
    # Test Queries (User se aane wali)
    test_queries = [
        "Meri allergy kya hai?",
        "Panadol aur Disprin saath le sakte hain?",
        "Diabetes ke symptoms kya hain?",
        "Meri last blood test kya thi?",
        "Asthma attack mein kya karein?",
    ]
    
    for q in test_queries:
        print(f"\n👤 User: {q}")
        print("🤖 Bot: ", end="")
        answer = process_query(q)
        print(answer)
        print("-"*50)
