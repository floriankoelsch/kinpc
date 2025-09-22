import os, json, requests, sys
from dotenv import load_dotenv

print("Arbeitsverzeichnis:", os.getcwd())

# 1) .env gezielt laden
dotenv_paths = [
    os.path.join(os.getcwd(), ".env"),
    os.path.join(os.path.dirname(__file__), ".env"),
]
loaded = False
for p in dotenv_paths:
    if os.path.isfile(p):
        print("Lade .env von:", p)
        load_dotenv(p, override=True)
        loaded = True
        break
if not loaded:
    print("WARNUNG: Keine .env gefunden an:", dotenv_paths)

# 2) Keys lesen + maskiert ausgeben
openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
eleven_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
voice_id   = (os.getenv("ELEVENLABS_VOICE_ID") or "").strip()

def mask(s):
    return f"{s[:4]}…{s[-4:]}" if s and len(s) > 8 else s

print("OPENAI_API_KEY:", mask(openai_key), "len=", len(openai_key))
print("ELEVENLABS_API_KEY:", mask(eleven_key), "len=", len(eleven_key))
print("ELEVENLABS_VOICE_ID:", voice_id or "(leer)")

if not eleven_key:
    print("\n❌ ELEVENLABS_API_KEY ist leer. Prüfe .env (keine Anführungszeichen, keine Leerzeichen).")
    sys.exit(1)

# 3) /v1/user testen
try:
    r = requests.get("https://api.elevenlabs.io/v1/user", headers={"xi-api-key": eleven_key}, timeout=20)
    print("GET /v1/user ->", r.status_code)
    print("Body:", r.text[:400])
    if r.status_code == 401:
        print("\n❌ 401 Unauthorized: Key falsch/abgelaufen/leer oder Encoding-Problem (.env).")
        print("   -> Regeneriere den API-Key im ElevenLabs-Dashboard und ersetze ihn in .env.")
        sys.exit(1)
except Exception as e:
    print("HTTP-Fehler:", e)
    sys.exit(1)

# 4) Voices abrufen
r = requests.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": eleven_key}, timeout=30)
print("\nGET /v1/voices ->", r.status_code)
if r.ok:
    voices = r.json().get("voices", [])
    print(f"Gefundene Voices: {len(voices)}")
    if voices:
        print("Beispiel-ID:", voices[0].get("voice_id"), "| Name:", voices[0].get("name"))
    else:
        print("WARN: Keine Voices im Account gelistet.")
else:
    print("Body:", r.text[:400])
