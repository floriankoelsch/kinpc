# npc_bot_proximity.py
import os, time, threading, io, sqlite3
from typing import Dict, List, Any, Optional, Tuple
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
load_dotenv()

# ----------------- Basis-Konfig -----------------
HOST = os.getenv("BOT_HOST", "127.0.0.1")
PORT = int(os.getenv("BOT_PORT", "9000"))

GREET_RADIUS = float(os.getenv("GREET_RADIUS", "4.0"))      # Begrüßen/Chat ab 4.0m
TARGET_SR = int(os.getenv("TARGET_SR", "48000"))
COOLDOWN_SEC = float(os.getenv("COOLDOWN_SEC", "20"))       # Mindestabstand zw. Events (Sicherheitsnetz)

# Gesprächslogik
LISTEN_DURATION_SEC = float(os.getenv("LISTEN_DURATION_SEC", "5.0"))   # wie lange "lauschen"
CONVO_INACTIVITY_TIMEOUT = float(os.getenv("CONVO_TIMEOUT_SEC", "90")) # wenn so lange kein Kontakt: Session reset
INTER_LISTEN_GAP = float(os.getenv("INTER_LISTEN_GAP_SEC", "2.0"))     # minimale Pause zwischen zwei "listen" Zyklen

# Audio-Routing:
# Output -> CABLE Input (Index 23 bevorzugt), .env OUT_DEVICE_ID übersteuert
OUT_DEVICE_ID_PREF = 23
OUT_DEVICE_ID_ENV = os.getenv("OUT_DEVICE_ID", "").strip()
# Input -> CABLE Output (per Name), .env IN_DEVICE_ID übersteuert
IN_DEVICE_ID_ENV = os.getenv("IN_DEVICE_ID", "").strip()
IN_DEVICE_NAME_PREFS = ["CABLE Output", "VB-Audio Point", "VoiceMeeter"]

PLAYBACK_BLOCKING = False

# OpenAI + ElevenLabs
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")

# ----------------- libs (robust) ----------------
_errs: List[str] = []
try:
    import numpy as np
except Exception as e:
    np = None; _errs.append(f"numpy fehlt: {e}")
try:
    import sounddevice as sd
    import soundfile as sf
except Exception as e:
    sd = None; sf = None; _errs.append(f"sounddevice/soundfile fehlt: {e}")
try:
    import requests
except Exception as e:
    requests = None; _errs.append(f"requests fehlt: {e}")

# ----------------- App/State --------------------
app = FastAPI(title="KI-NPC Bot")

positions_lock = threading.Lock()
npc_pos = [10.0, 10.0]   # NPC bleibt fix
players: List[Dict[str, Any]] = []

# per-player Zustand
# { player_id: { "mode": "idle"|"greeted"|"chatting", "last_seen": ts, "last_listen": ts, "last_interaction": ts } }
pstate: Dict[str, Dict[str, float]] = {}
last_greet_time: Dict[str, float] = {}  # Sicherheitsnetz (nicht für Logik nötig, aber ok)

log_lines: List[str] = []
_resolved_out: Optional[int] = None
_resolved_in: Optional[int] = None

# ----------------- Mini-Logger ------------------
def log(msg: str):
    ts = time.strftime("%H:%M:%S"); entry = f"[{ts}] {msg}"
    print(entry); log_lines.append(entry)
    if len(log_lines) > 500: del log_lines[:250]

def dist2(a, b): dx=a[0]-b[0]; dy=a[1]-b[1]; return (dx*dx+dy*dy)**0.5

# ----------------- SQLite Memory ----------------
DB_PATH = os.getenv("NPC_DB", "npc_memory.sqlite3")
db_lock = threading.Lock()

def db_init():
    with db_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
              player_id TEXT,
              ts REAL,
              role TEXT,         -- 'system' | 'user' | 'assistant'
              content TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_player_ts ON messages(player_id, ts)")
        conn.commit(); conn.close()

def db_add(player_id: str, role: str, content: str):
    with db_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cur = conn.cursor()
        cur.execute("INSERT INTO messages(player_id, ts, role, content) VALUES (?, ?, ?, ?)",
                    (player_id, time.time(), role, content))
        conn.commit(); conn.close()

def db_get_last(player_id: str, limit: int = 20) -> List[Tuple[str, str]]:
    with db_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cur = conn.cursor()
        cur.execute("SELECT role, content FROM messages WHERE player_id=? ORDER BY ts DESC LIMIT ?", (player_id, limit))
        rows = cur.fetchall(); conn.close()
    rows.reverse()
    return rows

def db_reset(player_id: str):
    with db_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE player_id=?", (player_id,))
        conn.commit(); conn.close()

db_init()

# ----------------- OpenAI/ElevenLabs -----------
def openai_chat(messages: List[Dict[str, str]], max_tokens=128, temp=0.7) -> str:
    if not OPENAI_API_KEY or requests is None:
        log("OpenAI nicht konfiguriert – Fallback-Text.")
        return "Okay."
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        data = {"model": OPENAI_MODEL, "messages": messages, "temperature": temp, "max_tokens": max_tokens}
        r = requests.post(url, headers=headers, json=data, timeout=12); r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log(f"OpenAI-Fehler: {e}"); return "Alles klar."

def elevenlabs_tts(text: str) -> Optional[bytes]:
    if not ELEVENLABS_API_KEY or requests is None:
        log("ElevenLabs nicht konfiguriert – kein Audio."); return None
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {"xi-api-key": ELEVENLABS_API_KEY, "accept":"audio/mpeg", "Content-Type":"application/json"}
        payload = {"text":text, "model_id":"eleven_multilingual_v2",
                   "voice_settings":{"stability":0.4,"similarity_boost":0.7}}
        r = requests.post(url, headers=headers, json=payload, timeout=15); r.raise_for_status()
        return r.content
    except Exception as e:
        log(f"ElevenLabs-Fehler: {e}"); return None

# ----------------- Audio I/O --------------------
def resolve_output_device() -> Optional[int]:
    if sd is None: return None
    try:
        devs = sd.query_devices()
        if OUT_DEVICE_ID_ENV:
            try:
                idx = int(OUT_DEVICE_ID_ENV); _ = devs[idx]
                log(f"Audio OUT: .env -> {idx} ({devs[idx]['name']})"); return idx
            except Exception:
                log(f"Audio OUT: ungültig .env '{OUT_DEVICE_ID_ENV}' – ignoriere.")
        try:
            _ = devs[OUT_DEVICE_ID_PREF]
            log(f"Audio OUT: feste Prio -> {OUT_DEVICE_ID_PREF} ({devs[OUT_DEVICE_ID_PREF]['name']})")
            return OUT_DEVICE_ID_PREF
        except Exception:
            pass
        for i, d in enumerate(devs):
            if d.get("max_output_channels",0) > 0 and any(p.lower() in str(d.get("name","")).lower() for p in ["CABLE Input","VB-Audio","VoiceMeeter"]):
                log(f"Audio OUT: auto -> {i} ({d['name']})"); return i
        d = sd.query_devices(None, "output"); log(f"Audio OUT: default -> {d['index']} ({d['name']})"); return d["index"]
    except Exception as e:
        log(f"Audio OUT: Query-Fehler: {e}"); return None

def resolve_input_device() -> Optional[int]:
    if sd is None: return None
    try:
        devs = sd.query_devices()
        if IN_DEVICE_ID_ENV:
            try:
                idx = int(IN_DEVICE_ID_ENV); _ = devs[idx]
                log(f"Audio IN: .env -> {idx} ({devs[idx]['name']})"); return idx
            except Exception:
                log(f"Audio IN: ungültig .env '{IN_DEVICE_ID_ENV}' – ignoriere.")
        for i, d in enumerate(devs):
            if d.get("max_input_channels",0) > 0:
                name = str(d.get("name",""))
                if any(p.lower() in name.lower() for p in IN_DEVICE_NAME_PREFS):
                    log(f"Audio IN: auto -> {i} ({name})"); return i
        d = sd.query_devices(None, "input"); log(f"Audio IN: default -> {d['index']} ({d['name']})"); return d["index"]
    except Exception as e:
        log(f"Audio IN: Query-Fehler: {e}"); return None

def play_audio_bytes_mp3(mp3_bytes: bytes):
    if sd is None or sf is None:
        log("Audio-Playback nicht verfügbar (sounddevice/soundfile fehlt)."); return
    try:
        with sf.SoundFile(io.BytesIO(mp3_bytes)) as f:
            data = f.read(dtype="float32"); sr = f.samplerate
        if sr != TARGET_SR and np is not None:
            import math
            ratio = TARGET_SR / sr
            new_len = int(math.ceil(len(data) * ratio))
            idx = np.linspace(0, len(data)-1, new_len, dtype=np.float32)
            data = np.interp(idx, np.arange(len(data), dtype=np.float32), data).astype("float32")
            sr = TARGET_SR
        global _resolved_out
        if _resolved_out is None: _resolved_out = resolve_output_device()
        if _resolved_out is not None: sd.default.device = (None, _resolved_out)
        log(f"Audio: play sr={sr} frames={len(data)} device={_resolved_out if _resolved_out is not None else 'default'}")
        sd.play(data, sr, blocking=PLAYBACK_BLOCKING); 
        if not PLAYBACK_BLOCKING: sd.wait()
    except Exception as e:
        log(f"Audio-Fehler: {e}")

def record_input_wav_bytes(seconds: float) -> Optional[bytes]:
    if sd is None or sf is None:
        log("Audio-Record nicht verfügbar (sounddevice/soundfile fehlt)."); return None
    try:
        global _resolved_in
        if _resolved_in is None: _resolved_in = resolve_input_device()
        if _resolved_in is not None: sd.default.device = (_resolved_in, None)
        channels = 1
        log(f"Audio: record {seconds:.1f}s @ {TARGET_SR}Hz from device={_resolved_in if _resolved_in is not None else 'default'}")
        rec = sd.rec(int(seconds * TARGET_SR), samplerate=TARGET_SR, channels=channels, dtype="float32")
        sd.wait()
        buf = io.BytesIO()
        sf.write(buf, rec, TARGET_SR, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()
    except Exception as e:
        log(f"Record-Fehler: {e}"); return None

def transcribe_wav(wav_bytes: bytes) -> str:
    if not OPENAI_API_KEY or requests is None:
        log("STT: OpenAI-Key fehlt – keine Transkription."); return ""
    try:
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        files = {
            "file": ("speech.wav", wav_bytes, "audio/wav"),
            "model": (None, OPENAI_STT_MODEL),
            "language": (None, "de"),
            "response_format": (None, "json"),
        }
        r = requests.post(url, headers=headers, files=files, timeout=30); r.raise_for_status()
        text = r.json().get("text", "").strip()
        log(f"STT: '{text}'")
        return text
    except Exception as e:
        log(f"STT-Fehler: {e}"); return ""

# ----------------- Dialog-Engine ----------------
SYS_PROMPT = "Du bist ein freundlicher NPC in einem Spiel. Antworte kurz (1 Satz), natürlich und auf Deutsch. Beziehe dich auf das bisherige Gespräch, bleib hilfreich und nett."

def make_messages_from_memory(player_id: str, user_utterance: Optional[str]=None) -> List[Dict[str, str]]:
    msgs = [{"role":"system","content": SYS_PROMPT}]
    mem = db_get_last(player_id, limit=20)
    for role, content in mem:
        msgs.append({"role": "assistant" if role=="assistant" else "user", "content": content})
    if user_utterance:
        msgs.append({"role":"user","content": user_utterance})
    return msgs

def initial_greet(player_id: str, name: str):
    # Begrüßung nur EINMAL -> danach in "chatting"
    greet_prompt = f"Jemand namens {name} kommt in 4m Nähe. Sag eine sehr kurze, freundliche Begrüßung (max. 8 Wörter)."
    text = openai_chat([{"role":"system","content":"Sprich sehr kurz, freundlich, Deutsch."},
                        {"role":"user","content": greet_prompt}], max_tokens=24)
    log(f"Greet -> '{text}'")
    db_add(player_id, "assistant", text)         # Begrüßung in Memory
    audio = elevenlabs_tts(text or f"Hallo {name}!")
    if audio: play_audio_bytes_mp3(audio)

def listen_and_maybe_reply(player_id: str):
    # Lauschen
    wav = record_input_wav_bytes(LISTEN_DURATION_SEC)
    if not wav: return False
    user_text = transcribe_wav(wav)
    if not user_text: 
        return False
    db_add(player_id, "user", user_text)

    # Antworten mit Memory-Kontext
    msgs = make_messages_from_memory(player_id)
    reply = openai_chat(msgs, max_tokens=96)
    log(f"Reply -> '{reply}'")
    db_add(player_id, "assistant", reply)
    audio2 = elevenlabs_tts(reply or "Alles klar.")
    if audio2: play_audio_bytes_mp3(audio2)
    return True

# ----------------- HTTP-API ---------------------
@app.get("/health")
def health(): return {"ok": True, "bot": "proximity"}

@app.get("/state")
def state():
    with positions_lock:
        return {"npc":{"x":npc_pos[0],"y":npc_pos[1]},
                "players":players,
                "radius":GREET_RADIUS,
                "sr":TARGET_SR,
                "resolved_out": _resolved_out,
                "resolved_in": _resolved_in,
                "lib_errors": _errs,
                "openai_model": OPENAI_MODEL,
                "stt_model": OPENAI_STT_MODEL,
                "listen_sec": LISTEN_DURATION_SEC,
                "sessions": pstate}

@app.get("/log")
def get_log(): return {"lines": log_lines[-160:]}

@app.get("/devices")
def devices():
    if sd is None: return {"error":"sounddevice nicht installiert"}
    try:
        devs = sd.query_devices()
        return {"devices": devs, 
                "default_output": sd.query_devices(None,"output"),
                "default_input": sd.query_devices(None,"input")}
    except Exception as e:
        return {"error": str(e)}

@app.get("/memory/{player_id}")
def memory(player_id: str):
    return {"player_id": player_id, "last20": db_get_last(player_id, 20)}

@app.post("/reset/{player_id}")
def reset(player_id: str):
    db_reset(player_id)
    pstate.pop(player_id, None)
    last_greet_time.pop(player_id, None)
    return {"ok": True}

@app.post("/set_out/{idx}")
def set_out(idx: int):
    global _resolved_out
    if sd is None: return {"error":"sounddevice nicht installiert"}
    try:
        dev = sd.query_devices()[idx]
        _resolved_out = idx
        log(f"Audio OUT: manuell -> {idx} ({dev['name']})")
        return {"ok": True, "device": dev}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/set_in/{idx}")
def set_in(idx: int):
    global _resolved_in
    if sd is None: return {"error":"sounddevice nicht installiert"}
    try:
        dev = sd.query_devices()[idx]
        _resolved_in = idx
        log(f"Audio IN: manuell -> {idx} ({dev['name']})")
        return {"ok": True, "device": dev}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/update")
async def update(req: Request):
    """
    payload = {
      "npc": {"x": float, "y": float},   # wird ignoriert (NPC bleibt fest)
      "players": [{"id": "...", "name": "...", "x": float, "y": float}, ...]
    }
    """
    data = await req.json()
    ps = data.get("players", [])
    if not isinstance(ps, list):
        return JSONResponse({"error":"Invalid payload"}, status_code=400)
    with positions_lock:
        # NPC-Position bleibt unverändert
        norm = []
        for p in ps:
            try:
                norm.append({"id": str(p.get("id","p?")),
                             "name": (p.get("name") or p.get("id") or "Spieler"),
                             "x": float(p.get("x",0.0)), "y": float(p.get("y",0.0))})
            except: pass
        players.clear(); players.extend(norm)
    return {"ok": True}

# ----------------- Nähe-Loop --------------------
def proximity_loop():
    log("Proximity-Loop gestartet (Dialog-Modus, radius=%.2f)." % GREET_RADIUS)
    while True:
        try:
            now = time.time()
            with positions_lock:
                npc = (npc_pos[0], npc_pos[1]); ps = list(players)
            for p in ps:
                name = p.get("name","Spieler")
                pid  = p.get("id", name)
                d = dist2(npc, (p["x"], p["y"]))
                st = pstate.get(pid, {"mode":"idle", "last_seen":0, "last_listen":0, "last_interaction":0})
                # Update last_seen immer, wenn in Reichweite
                if d <= GREET_RADIUS:
                    st["last_seen"] = now
                    # Session-Start?
                    if st["mode"] == "idle":
                        # kein erneutes Begrüßen wenn kurz vorher bereits gegrüßt (Sicherheitsnetz)
                        if (now - last_greet_time.get(pid, 0)) >= 3.0:
                            last_greet_time[pid] = now
                            log(f"[SESSION] start -> {pid} ({name}) dist={d:.2f}")
                            initial_greet(pid, name)
                        st["mode"] = "chatting"
                        st["last_interaction"] = now
                        pstate[pid] = st
                        continue

                    # Chatting: zyklisch lauschen -> wenn Spieler was sagt, antworten
                    if st["mode"] == "chatting":
                        if (now - st.get("last_listen", 0)) >= INTER_LISTEN_GAP:
                            st["last_listen"] = now
                            pstate[pid] = st
                            said = listen_and_maybe_reply(pid)
                            if said:
                                st["last_interaction"] = time.time()
                                pstate[pid] = st
                else:
                    # Außerhalb des Radius: nichts tun, aber ggf. Timeout prüfen (weiter unten)
                    pass

                # Konversations-Timeout
                if st["mode"] != "idle":
                    idle_for = now - max(st.get("last_seen",0), st.get("last_interaction",0))
                    if idle_for >= CONVO_INACTIVITY_TIMEOUT:
                        log(f"[SESSION] end -> {pid} idle {idle_for:.1f}s")
                        st["mode"] = "idle"
                        pstate[pid] = st
                        # Memory bleibt erhalten (gewollt)

        except Exception as e:
            log(f"Loop-Fehler: {e}")
        time.sleep(0.2)

def main():
    t = threading.Thread(target=proximity_loop, daemon=True); t.start()
    import uvicorn
    log(f"HTTP: POST http://{HOST}:{PORT}/update | GET /state /log /devices /memory/{{id}} /reset/{{id}} /set_out/{{i}} /set_in/{{i}}")
    uvicorn.run(app, host=HOST, port=PORT, reload=False)

if __name__ == "__main__":
    main()
