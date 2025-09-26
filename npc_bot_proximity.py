# npc_bot_proximity.py
import os, time, threading, io, sqlite3
from typing import Dict, List, Any, Optional, Tuple
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
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

# NPC-Konfiguration
NPC1_NAME = os.getenv("NPC1_NAME", "NPC 1")
NPC2_NAME = os.getenv("NPC2_NAME", "NPC 2")
NPC_CHAT_GAP_SEC = float(os.getenv("NPC_CHAT_GAP_SEC", "8.0"))

def _persona_default(name: str, role: str) -> str:
    return (
        f"Du bist {name}, ein {role} in einer futuristischen Stadt. "
        "Sprich locker, freundlich und auf Deutsch. Reagiere kurz (1-2 Sätze) und geh auf die letzten Aussagen ein."
    )

NPC_CONFIG: Dict[str, Dict[str, str]] = {
    "npc1": {
        "name": NPC1_NAME,
        "persona": os.getenv(
            "NPC1_PERSONA",
            _persona_default(NPC1_NAME, "neugieriger Stadtführer, der Besucher willkommen heißt"),
        ),
        "voice_id": os.getenv("ELEVENLABS_VOICE_ID_NPC1", ELEVENLABS_VOICE_ID),
    },
    "npc2": {
        "name": NPC2_NAME,
        "persona": os.getenv(
            "NPC2_PERSONA",
            _persona_default(NPC2_NAME, "entspannter Techniker, der gern Anekdoten erzählt"),
        ),
        "voice_id": os.getenv("ELEVENLABS_VOICE_ID_NPC2", os.getenv("ELEVENLABS_ALT_VOICE_ID", ELEVENLABS_VOICE_ID)),
    },
}

NPC_IDS = list(NPC_CONFIG.keys())
NPC_PAIR_KEY = "npcpair:" + ":".join(NPC_IDS)
PAIR_DB_ID = NPC_PAIR_KEY

environment_profile: Dict[str, Any] = {
    "id": None,
    "name": "",
    "radius": GREET_RADIUS,
    "weights": {},
    "task": "",
    "agents": [],
}

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
npc_positions: Dict[str, List[float]] = {
    "npc1": [10.0, 10.0],
    "npc2": [14.0, 10.0],
}
players: List[Dict[str, Any]] = []

# per-player Zustand
# { player_id: { "mode": "idle"|"chatting", "last_seen": ts, "last_listen": ts, "last_interaction": ts } }
pstate: Dict[str, Dict[str, float]] = {}
npc_pair_state: Dict[str, Dict[str, Any]] = {
    NPC_PAIR_KEY: {
        "mode": "idle",
        "last_exchange": 0.0,
        "next_speaker": "npc1",
        "player_last_greet": {},
    }
}

log_lines: List[str] = []
_resolved_out: Optional[int] = None
_resolved_in: Optional[int] = None


def _jsonable(obj: Any) -> Any:
    """Convert sounddevice structures to JSON-friendly values."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    # numpy scalars provide ``item`` to retrieve Python primitives
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj

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

# ----------------- Environment Sync -----------
def apply_environment_metadata(env_data: Optional[Dict[str, Any]]) -> None:
    global GREET_RADIUS
    if not isinstance(env_data, dict):
        return
    env_id = env_data.get("id")
    environment_profile["id"] = env_id
    environment_profile["name"] = env_data.get("name", environment_profile.get("name", ""))
    environment_profile["weights"] = env_data.get("weights", environment_profile.get("weights", {}))
    environment_profile["task"] = env_data.get("task", environment_profile.get("task", ""))
    environment_profile["agents"] = env_data.get("agents", environment_profile.get("agents", []))
    radius = env_data.get("radius_m")
    if radius is not None:
        try:
            new_radius = min(4.0, float(radius))
            environment_profile["radius"] = new_radius
            GREET_RADIUS = new_radius
        except (TypeError, ValueError):
            pass
    agents = env_data.get("agents", [])
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        aid = agent.get("id")
        if not aid:
            continue
        cfg = NPC_CONFIG.setdefault(
            aid,
            {
                "name": aid,
                "persona": _persona_default(aid, "NPC"),
                "voice_id": ELEVENLABS_VOICE_ID,
            },
        )
        if agent.get("name"):
            cfg["name"] = agent["name"]
        if agent.get("prompt"):
            cfg["persona"] = agent["prompt"]
        if agent.get("voice_id"):
            cfg["voice_id"] = agent["voice_id"]

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

def elevenlabs_tts(text: str, voice_id: Optional[str] = None) -> Optional[bytes]:
    if not ELEVENLABS_API_KEY or requests is None:
        log("ElevenLabs nicht konfiguriert – kein Audio."); return None
    try:
        vid = (voice_id or ELEVENLABS_VOICE_ID or "").strip() or ELEVENLABS_VOICE_ID
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
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
def ensure_pair_state(pair_key: str) -> Dict[str, Any]:
    st = npc_pair_state.setdefault(
        pair_key,
        {"mode": "idle", "last_exchange": 0.0, "next_speaker": NPC_IDS[0], "player_last_greet": {}},
    )
    st.setdefault("player_last_greet", {})
    if st.get("next_speaker") not in NPC_IDS:
        st["next_speaker"] = NPC_IDS[0]
    return st

def other_npc(current: str) -> str:
    for nid in NPC_IDS:
        if nid != current:
            return nid
    return current

def transcript_from_memory(limit: int = 40) -> str:
    rows = db_get_last(PAIR_DB_ID, limit=limit)
    lines: List[str] = []
    for role, content in rows:
        if role.startswith("player:"):
            name = role.split(":", 1)[1] or "Spieler"
            lines.append(f"Spieler {name}: {content}")
        else:
            cfg = NPC_CONFIG.get(role, {"name": role})
            lines.append(f"{cfg.get('name', role)}: {content}")
    return "\n".join(lines)

def npc_pair_say(pair_key: str, speaker: str, focus: Optional[str] = None) -> Optional[str]:
    cfg = NPC_CONFIG.get(speaker)
    if not cfg:
        return None
    transcript = transcript_from_memory()
    prompt_parts: List[str] = []
    if transcript:
        prompt_parts.append("Bisheriger Dialog:\n" + transcript)
    else:
        prompt_parts.append("Das Gespräch beginnt gerade – starte locker und natürlich.")
    if focus:
        prompt_parts.append(f"Hinweis: {focus}")
    prompt_parts.append(f"Antworte als {cfg['name']} mit 1-2 kurzen Sätzen auf Deutsch.")
    messages = [
        {"role": "system", "content": cfg["persona"]},
        {"role": "user", "content": "\n\n".join(prompt_parts)},
    ]
    reply = openai_chat(messages, max_tokens=120)
    if not reply:
        reply = "Alles klar."
    log(f"{cfg['name']} -> '{reply}'")
    db_add(PAIR_DB_ID, speaker, reply)
    audio = elevenlabs_tts(reply, voice_id=cfg.get("voice_id"))
    if audio:
        play_audio_bytes_mp3(audio)
    st = ensure_pair_state(pair_key)
    st["last_exchange"] = time.time()
    st["mode"] = "chatting"
    st["next_speaker"] = other_npc(speaker)
    npc_pair_state[pair_key] = st
    return reply

def npc_pair_start(pair_key: str):
    st = ensure_pair_state(pair_key)
    if st.get("mode") == "chatting":
        return
    log("[NPC-PAIR] Gespräch startet")
    st["mode"] = "chatting"
    st["next_speaker"] = NPC_IDS[0]
    npc_pair_state[pair_key] = st
    npc_pair_say(pair_key, NPC_IDS[0], focus=f"Beginne das Gespräch mit {NPC_CONFIG[NPC_IDS[1]]['name']}.")
    npc_pair_say(pair_key, NPC_IDS[1], focus=f"Reagiere direkt auf {NPC_CONFIG[NPC_IDS[0]]['name']} und halte das Gespräch in Gang.")

def npc_pair_pause(pair_key: str, reason: str = "zu weit entfernt"):
    st = ensure_pair_state(pair_key)
    if st.get("mode") != "idle":
        log(f"[NPC-PAIR] Gespräch pausiert ({reason})")
    st["mode"] = "idle"
    npc_pair_state[pair_key] = st

def npc_pair_greet_player(pair_key: str, player_id: str, player_name: str):
    st = ensure_pair_state(pair_key)
    speaker = st.get("next_speaker") or NPC_IDS[0]
    npc_pair_say(pair_key, speaker, focus=f"Begrüße freundlich den Spieler namens {player_name}, der gerade dazu stößt.")
    st = ensure_pair_state(pair_key)
    st.setdefault("player_last_greet", {})[player_id] = time.time()
    npc_pair_state[pair_key] = st

def handle_player_speech(pair_key: str, player_id: str, player_name: str) -> bool:
    wav = record_input_wav_bytes(LISTEN_DURATION_SEC)
    if not wav:
        return False
    user_text = transcribe_wav(wav)
    if not user_text:
        return False
    db_add(PAIR_DB_ID, f"player:{player_id}", user_text)
    st = ensure_pair_state(pair_key)
    speaker = st.get("next_speaker") or NPC_IDS[0]
    focus = f"Der Spieler {player_name} sagt: '{user_text}'. Antworte direkt darauf."
    npc_pair_say(pair_key, speaker, focus=focus)
    return True

# ----------------- HTTP-API ---------------------
@app.get("/health")
def health(): return {"ok": True, "bot": "proximity"}

@app.get("/state")
def state():
    with positions_lock:
        npc_list = [
            {"id": nid, "name": NPC_CONFIG.get(nid, {}).get("name", nid), "x": pos[0], "y": pos[1]}
            for nid, pos in npc_positions.items()
        ]
        pl = list(players)
    pair_info = ensure_pair_state(NPC_PAIR_KEY)
    return {
        "npcs": npc_list,
        "players": pl,
        "radius": GREET_RADIUS,
        "sr": TARGET_SR,
        "resolved_out": _resolved_out,
        "resolved_in": _resolved_in,
        "lib_errors": _errs,
        "openai_model": OPENAI_MODEL,
        "stt_model": OPENAI_STT_MODEL,
        "listen_sec": LISTEN_DURATION_SEC,
        "sessions": pstate,
        "pair_state": pair_info,
        "environment": environment_profile,
    }

@app.get("/log")
def get_log(): return {"lines": log_lines[-160:]}

@app.get("/devices")
def devices():
    if sd is None:
        return {"error": "sounddevice nicht installiert"}
    try:
        devs = sd.query_devices()
        payload = {
            "devices": [_jsonable(dict(d)) for d in devs],
            "default_output": _jsonable(dict(sd.query_devices(None, "output"))),
            "default_input": _jsonable(dict(sd.query_devices(None, "input"))),
        }
        return JSONResponse(content=jsonable_encoder(payload))
    except Exception as e:
        return {"error": str(e)}

@app.get("/memory/{player_id}")
def memory(player_id: str):
    return {"player_id": player_id, "last20": db_get_last(player_id, 20)}

@app.post("/reset/{player_id}")
def reset(player_id: str):
    db_reset(player_id)
    pstate.pop(player_id, None)
    st = ensure_pair_state(NPC_PAIR_KEY)
    st.get("player_last_greet", {}).pop(player_id, None)
    npc_pair_state[NPC_PAIR_KEY] = st
    return {"ok": True}

@app.post("/set_out/{idx}")
def set_out(idx: int):
    global _resolved_out
    if sd is None:
        return {"error": "sounddevice nicht installiert"}
    try:
        dev = dict(sd.query_devices()[idx])
        _resolved_out = idx
        log(f"Audio OUT: manuell -> {idx} ({dev['name']})")
        return JSONResponse(content=jsonable_encoder({"ok": True, "device": _jsonable(dev)}))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/set_in/{idx}")
def set_in(idx: int):
    global _resolved_in
    if sd is None:
        return {"error": "sounddevice nicht installiert"}
    try:
        dev = dict(sd.query_devices()[idx])
        _resolved_in = idx
        log(f"Audio IN: manuell -> {idx} ({dev['name']})")
        return JSONResponse(content=jsonable_encoder({"ok": True, "device": _jsonable(dev)}))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/update")
async def update(req: Request):
    """
    payload = {
      "npcs": [{"id": "npc1", "x": float, "y": float}, ...],
      "players": [{"id": "...", "name": "...", "x": float, "y": float}, ...]
    }
    """
    data = await req.json()
    ps = data.get("players", [])
    npcs_in = data.get("npcs", [])
    apply_environment_metadata(data.get("environment"))
    if not isinstance(ps, list):
        return JSONResponse({"error":"Invalid payload"}, status_code=400)
    with positions_lock:
        if isinstance(npcs_in, list):
            for n in npcs_in:
                try:
                    nid = str(n.get("id", ""))
                    x_val = float(n.get("x", 0.0))
                    y_val = float(n.get("y", 0.0))
                    if nid in npc_positions:
                        npc_positions[nid][0] = x_val
                        npc_positions[nid][1] = y_val
                    else:
                        npc_positions[nid] = [x_val, y_val]
                except Exception:
                    pass
        norm = []
        for p in ps:
            try:
                norm.append({"id": str(p.get("id","p?")),
                             "name": (p.get("name") or p.get("id") or "Spieler"),
                             "x": float(p.get("x",0.0)), "y": float(p.get("y",0.0))})
            except Exception:
                pass
        players.clear(); players.extend(norm)
    return {"ok": True}

# ----------------- Nähe-Loop --------------------
def proximity_loop():
    log("Proximity-Loop gestartet (Dialog-Modus, radius=%.2f)." % GREET_RADIUS)
    while True:
        try:
            now = time.time()
            with positions_lock:
                npc_copy = {nid: tuple(pos) for nid, pos in npc_positions.items()}
                ps = list(players)
            npc1 = npc_copy.get("npc1", (0.0, 0.0))
            npc2 = npc_copy.get("npc2", (0.0, 0.0))

            player_entries = []
            any_player_within_radius = False
            for p in ps:
                name = p.get("name", "Spieler")
                pid = p.get("id", name)
                st = pstate.get(pid)
                if st is None:
                    st = {"mode": "idle", "last_seen": 0.0, "last_listen": 0.0, "last_interaction": 0.0}
                    pstate[pid] = st
                player_pos = (p.get("x", 0.0), p.get("y", 0.0))
                min_dist = min(dist2(npc1, player_pos), dist2(npc2, player_pos))
                if min_dist <= GREET_RADIUS:
                    any_player_within_radius = True
                player_entries.append((pid, name, st, min_dist))

            pair_state = ensure_pair_state(NPC_PAIR_KEY)
            dist_pair = dist2(npc1, npc2)
            if dist_pair <= GREET_RADIUS and any_player_within_radius:
                if pair_state.get("mode") != "chatting":
                    npc_pair_start(NPC_PAIR_KEY)
                    pair_state = ensure_pair_state(NPC_PAIR_KEY)
                elif (now - pair_state.get("last_exchange", 0)) >= NPC_CHAT_GAP_SEC:
                    next_speaker = pair_state.get("next_speaker") or NPC_IDS[0]
                    npc_pair_say(NPC_PAIR_KEY, next_speaker)
                    pair_state = ensure_pair_state(NPC_PAIR_KEY)
            else:
                reason = "zu weit entfernt" if dist_pair > GREET_RADIUS else "keine Spieler in Reichweite"
                npc_pair_pause(NPC_PAIR_KEY, reason=reason)
                pair_state = ensure_pair_state(NPC_PAIR_KEY)

            pair_active = pair_state.get("mode") == "chatting"
            for pid, name, st, min_dist in player_entries:
                if min_dist <= GREET_RADIUS and pair_active:
                    st["last_seen"] = now
                    if st["mode"] == "idle":
                        last_greet = ensure_pair_state(NPC_PAIR_KEY)["player_last_greet"].get(pid, 0)
                        if (now - last_greet) >= 5.0:
                            log(f"[PLAYER] join -> {pid} dist={min_dist:.2f}")
                            npc_pair_greet_player(NPC_PAIR_KEY, pid, name)
                        st["mode"] = "chatting"
                        st["last_interaction"] = now
                        pstate[pid] = st
                        continue

                    if st["mode"] == "chatting" and (now - st.get("last_listen", 0)) >= INTER_LISTEN_GAP:
                        st["last_listen"] = now
                        pstate[pid] = st
                        said = handle_player_speech(NPC_PAIR_KEY, pid, name)
                        if said:
                            st["last_interaction"] = time.time()
                            pstate[pid] = st
                else:
                    if st["mode"] != "idle":
                        idle_for = now - max(st.get("last_seen", 0), st.get("last_interaction", 0))
                        if idle_for >= CONVO_INACTIVITY_TIMEOUT:
                            log(f"[PLAYER] end -> {pid} idle {idle_for:.1f}s")
                            st["mode"] = "idle"
                            pstate[pid] = st

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
