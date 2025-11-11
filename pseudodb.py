import os
import json
from datetime import datetime

# ✅ Path for pseudo database file
DB_FILE = os.path.join(os.path.dirname(__file__), "pseudodb.json")

# ✅ Initialize pseudodb file if not exists
def init_pseudodb():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w") as f:
            json.dump({"users": [], "transcriptions": []}, f)
    else:
        # Ensure valid structure
        try:
            with open(DB_FILE, "r") as f:
                data = json.load(f)
            if "users" not in data or "transcriptions" not in data:
                raise ValueError
        except Exception:
            with open(DB_FILE, "w") as f:
                json.dump({"users": [], "transcriptions": []}, f)

def _load():
    with open(DB_FILE, "r") as f:
        return json.load(f)

def _save(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ✅ User Management
def add_user(username, email, password_hash):
    data = _load()
    user = {"id": len(data["users"]) + 1, "username": username, "email": email, "password_hash": password_hash}
    data["users"].append(user)
    _save(data)
    return user

def find_user_by_email_or_username(identifier):
    data = _load()
    for u in data["users"]:
        if u["email"] == identifier or u["username"] == identifier:
            return u
    return None

# ✅ Transcription Management
def add_transcription(user_id, filename, text, language):
    data = _load()
    entry = {
        "id": len(data["transcriptions"]) + 1,
        "user_id": user_id,
        "filename": filename,
        "text": text,
        "language": language,
        "created_at": datetime.utcnow().isoformat()
    }
    data["transcriptions"].append(entry)
    _save(data)
    return entry

def list_transcriptions():
    data = _load()
    return data["transcriptions"]
