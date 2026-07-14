from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
import os

app = FastAPI()

@app.get("/")
def home():
    return "Healthcare Bot is running!"

@app.get("/patients")
def patients():
    return {"status": "ok"}

@app.post("/whatsapp")
async def whatsapp(request: Request):
    try:
        form = await request.form()
        msg = form.get('Body', '').strip()
        sender = form.get('From', '')
        print(f"📩 {sender}: {msg}")
        
        reply = f"Hello! You said: {msg}. I am a healthcare bot."
        
        resp = MessagingResponse()
        resp.message(reply)
        return PlainTextResponse(str(resp), media_type="application/xml")
    except Exception as e:
        return PlainTextResponse(f"Error: {e}", status_code=500)
