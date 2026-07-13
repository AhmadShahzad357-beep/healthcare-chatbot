import json
import re
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None
    print("⚠️ requests library nahi mili. OpenFDA feature disabled. Install: pip install requests")

# ================== Local Interaction Database ==================
# (Common medicines ke interactions — practice ke liye)
INTERACTION_DB = {
    ("aspirin", "warfarin"): {
        "severity": "High",
        "message": "Bleeding risk significantly increases. Avoid concurrent use without doctor supervision."
    },
    ("paracetamol", "aspirin"): {
        "severity": "Moderate",
        "message": "Both are painkillers. Taking together may increase side effects like stomach irritation."
    },
    ("paracetamol", "ibuprofen"): {
        "severity": "Moderate",
        "message": "Can be taken together in some cases, but consult a doctor for proper dosing intervals."
    },
    ("ibuprofen", "aspirin"): {
        "severity": "High",
        "message": "Combining NSAIDs increases risk of stomach ulcers and bleeding. Avoid."
    },
    ("lisinopril", "ibuprofen"): {
        "severity": "High",
        "message": "Ibuprofen can reduce the effectiveness of blood pressure medicines like Lisinopril."
    },
    ("metformin", "insulin"): {
        "severity": "Moderate",
        "message": "May increase risk of hypoglycemia (low blood sugar). Monitor blood sugar levels closely."
    },
    ("atenolol", "verapamil"): {
        "severity": "High",
        "message": "Combining these can cause severe bradycardia (slow heart rate) and heart block."
    },
}

# ================== Normalization Helper ==================
def _normalize_drug_name(name: str) -> str:
    """Medicine ka naam lowercase mein convert karein aur common synonyms map karein"""
    name = name.lower().strip()
    # Common synonyms
    synonyms = {
        "panadol": "paracetamol",
        "acetaminophen": "paracetamol",
        "brufen": "ibuprofen",
        "advil": "ibuprofen",
        "motrin": "ibuprofen",
        "disprin": "aspirin",
        "ecospirin": "aspirin",
        "aspirine": "aspirin",
    }
    return synonyms.get(name, name)

# ================== 1. Local Interaction Check ==================
def check_local_interaction(drug1: str, drug2: str) -> Optional[Dict]:
    """Local database se interaction check karein"""
    d1 = _normalize_drug_name(drug1)
    d2 = _normalize_drug_name(drug2)
    
    # Dono combinations check karein (A,B) ya (B,A)
    key1 = (d1, d2)
    key2 = (d2, d1)
    
    if key1 in INTERACTION_DB:
        return INTERACTION_DB[key1]
    elif key2 in INTERACTION_DB:
        return INTERACTION_DB[key2]
    else:
        return None

# ================== 2. OpenFDA Drug Info Lookup ==================
def fetch_drug_info_from_openfda(drug_name: str) -> Optional[str]:
    """
    OpenFDA API se drug ka general information fetch karein
    (e.g., uses, warnings)
    """
    if not requests:
        return None
    
    try:
        # OpenFDA drug label search
        url = "https://api.fda.gov/drug/label.json"
        params = {
            "search": f"openfda.brand_name:{drug_name}+OR+openfda.generic_name:{drug_name}",
            "limit": 1
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                result = data["results"][0]
                # Extract relevant info
                info = []
                if result.get("indications_and_usage"):
                    info.append(f"Indication: {result['indications_and_usage'][0][:200]}...")
                if result.get("warnings"):
                    info.append(f"Warning: {result['warnings'][0][:200]}...")
                if info:
                    return "\n".join(info)
        return None
    except Exception as e:
        print(f"⚠️ OpenFDA error: {e}")
        return None

# ================== 3. Main API Function ==================
def check_medicine_interaction(drug1: str, drug2: str) -> Dict:
    """
    Main function — pehle local DB check karega, phir OpenFDA (optional).
    """
    result = {
        "drug1": drug1,
        "drug2": drug2,
        "interaction_found": False,
        "severity": "Unknown",
        "message": "",
        "source": "Local DB"
    }
    
    # 1. Local check
    local_result = check_local_interaction(drug1, drug2)
    if local_result:
        result["interaction_found"] = True
        result["severity"] = local_result["severity"]
        result["message"] = local_result["message"]
        return result
    
    # 2. Agar local mein nahi mila, toh OpenFDA se generic info try karein
    d1_info = fetch_drug_info_from_openfda(drug1)
    d2_info = fetch_drug_info_from_openfda(drug2)
    
    if d1_info or d2_info:
        result["interaction_found"] = False
        result["message"] = f"No known major interaction found in local DB.\n\nInfo about {drug1}: {d1_info or 'Not found'}\n\nInfo about {drug2}: {d2_info or 'Not found'}"
        result["source"] = "Local DB + OpenFDA"
        return result
    
    # 3. Kuch nahi mila
    result["message"] = f"No known interaction data found for '{drug1}' and '{drug2}'. Please consult a doctor or pharmacist."
    return result

# ================== Testing ==================
if __name__ == "__main__":
    print("="*50)
    print("💊 Medicine API Test")
    print("="*50)
    
    test_cases = [
        ("Panadol", "Disprin"),
        ("Ibuprofen", "Aspirin"),
        ("Metformin", "Insulin"),
        ("Paracetamol", "Vitamin C"),
    ]
    
    for d1, d2 in test_cases:
        print(f"\n🔍 Checking: '{d1}' + '{d2}'")
        result = check_medicine_interaction(d1, d2)
        
        if result["interaction_found"]:
            print(f"   ⚠️ Interaction Found! Severity: {result['severity']}")
            print(f"   📝 {result['message']}")
        else:
            print(f"   ℹ️ {result['message'][:150]}...")
        print(f"   📍 Source: {result['source']}")
    
    print("\n✅ Medicine API test complete!")
