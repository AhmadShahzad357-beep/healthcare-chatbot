import re

class SafetyGuard:
    def __init__(self):
        pass

    def check_hallucination(self, response: str, context: str) -> bool:
        """
        Check if response is grounded in context.
        Returns True (Safe) or False (Hallucination).
        """
        # Agar LLM ne clearly "nahi pata" bola toh safe hai
        dont_know = ["mujhe nahi pata", "i don't know", "not found", "no information"]
        if any(phrase in response.lower() for phrase in dont_know):
            return True
        
        # Agar LLM ne source cite kiya toh safe hai
        if "source:" in response.lower() or "medlineplus" in response.lower():
            return True
        
        # Agar context empty hai aur LLM ne jawab de diya, toh hallucination
        if "no relevant information" in context.lower() or not context.strip():
            return False
            
        # Default: Safe (practice ke liye)
        return True
