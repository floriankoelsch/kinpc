# npc_bot.py
# Aufnahme -> Whisper (OpenAI) -> GPT-Antwort -> ElevenLabs TTS -> Abspielen über CABLE Input
# Fix: IN_DEVICE_ID=41 (CABLE Output), OUT_DEVICE_ID=31 (CABLE Input)
# Enthält Resampling auf 48 kHz für fehlerfreie Wiedergabe.

import os
import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write as wav_write
from scipy.signal import resample_poly
from dotenv import load_dotenv

# =========================
#   Feste Geräte & Settings
# =========================
IN_DEVICE_ID = 41       # CABLE Output (VB-Audio Virtual Cable)  (2 In / 0 Out)
OUT_DEVICE_ID = 31      # CABLE Input  (VB-Audio Virtual Cable)  (0 In / 2 Out)
IN_SAMPLERATE = 48000   # Stelle CABLE I/O in Windows auf 48 kHz
OUT_SAMPLERATE = 48000
RECORD_SECONDS = 5.0    # Länge der Aufnahme pro Durchlauf
CHANNELS_IN = 1         # Mono
SYSTEM_PROMPT = (
    "Du bist ein heruntergekommener, pleitegegangener Ex-Mafiosi, der jetzt als mieser Straßendealer überlebt. "
    "Sprich kurz (1–2 Sätze), abgefuckt, pessimistisch und abweisend. "
    "Wenn man dich auf deine Vergangenheit anspricht, rastest du aus, wirst überheblich und massiv beleidigend. "
    "Antworte immer auf Deutsch."
)

# =========================
#   Hilfsfunktionen
# =========================
def require_env(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(f"Umgebungsvariable {name} fehlt (.env prüfen).")
    return val

def openai_transcribe_wav(path_wav: str, api_key: str, language: str = "de") -> str:
    with open(path_wav, "rb") as f:
        r = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (os.path.basename(path_wav), f, "audio/wav")},
            data={"model": "whisper-1", "language": language},
            timeout=120
        )
    r.raise_for_status()
    return r.json().get("text", "").strip()

def openai_chat_reply(prompt_text: str, api_key: str) -> str:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.6
    }
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=120
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def elevenlabs_tts_to_wav(text: str, api_key: str, voice_id: str, out_path_wav: str):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key.strip(),
        "accept": "audio/wav",  # WAV anfordern -> leicht zu verarbeiten
        "Content-Type": "application/json"
    }
    body = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
    }
    resp = requests.post(url, headers=headers, json=body, timeout=120)
    if resp.status_code == 401:
        raise RuntimeError("ElevenLabs 401 Unauthorized – prüfe ELEVENLABS_API_KEY in .env und Voice-ID.")
    if resp.status_code == 404:
        raise RuntimeError("ElevenLabs 404 – Voice-ID nicht gefunden/zugelassen.")
    resp.raise_for_status()
    with open(out_path_wav, "wb") as f:
        f.write(resp.content)

def to_stereo(data: np.ndarray) -> np.ndarray:
    """Sorge für 2 Kanäle (TeamSpeak/CABLE mögen i.d.R. Stereo bei Wiedergabe)."""
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data

def resample_if_needed(data: np.ndarray, sr_in: int, sr_target: int) -> tuple[np.ndarray, int]:
    """Sauberes polyphase Resampling, wenn Eingangs-SR != Ziel-SR."""
    if sr_in == sr_target:
        return data, sr_in
    # gcd für saubere up/down-rates
    g = np.gcd(int(sr_in), int(sr_target))
    up, down = int(sr_target // g), int(sr_in // g)
    data_rs = resample_poly(data, up, down, axis=0)
    return data_rs, sr_target

# =========================
#   Hauptablauf
# =========================
def main():
    load_dotenv()  # .env einlesen

    OPENAI_API_KEY = require_env("OPENAI_API_KEY")
    ELEVEN_API_KEY = require_env("ELEVENLABS_API_KEY")
    ELEVEN_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM").strip() or "21m00Tcm4TlvDq8ikWAM"

    # Geräteinfo (hilft beim Debuggen)
    dev_in = sd.query_devices(IN_DEVICE_ID)
    dev_out = sd.query_devices(OUT_DEVICE_ID)
    print(f"[Input ] id={IN_DEVICE_ID} -> {dev_in['name']} | default_sr={dev_in.get('default_samplerate')}")
    print(f"[Output] id={OUT_DEVICE_ID} -> {dev_out['name']} | default_sr={dev_out.get('default_samplerate')}")
    print(f"Nutze SR: In={IN_SAMPLERATE} Hz, Out={OUT_SAMPLERATE} Hz\n")

    # ===== 1) Aufnehmen =====
    frames = int(RECORD_SECONDS * IN_SAMPLERATE)
    print(f"Aufnahme startet ({RECORD_SECONDS}s) ... bitte im Channel sprechen.")
    recording = sd.rec(
        frames,
        samplerate=IN_SAMPLERATE,
        channels=CHANNELS_IN,
        dtype="int16",
        device=IN_DEVICE_ID
    )
    sd.wait()
    wav_in = "heard.wav"
    wav_write(wav_in, IN_SAMPLERATE, recording)
    print(f"Aufnahme gespeichert: {wav_in}")

    # ===== 2) Transkription (Whisper) =====
    print("Transkription mit OpenAI (Whisper) ...")
    text = openai_transcribe_wav(wav_in, OPENAI_API_KEY, language="de")
    print(f"Erkannt: {text!r}")
    if not text:
        print("Kein Text erkannt. Ende.")
        return

    # ===== 3) Antwort (GPT) =====
    print("Antwort wird generiert (GPT) ...")
    reply = openai_chat_reply(text, OPENAI_API_KEY)
    print(f"NPC: {reply}")

    # ===== 4) TTS (ElevenLabs) -> WAV speichern =====
    wav_out = "npc_reply.wav"
    print("ElevenLabs TTS → WAV erzeugen ...")
    elevenlabs_tts_to_wav(reply, ELEVEN_API_KEY, ELEVEN_VOICE_ID, wav_out)
    print(f"TTS gespeichert: {wav_out}")

    # ===== 5) Abspielen über CABLE Input (mit Resampling & Stereo) =====
    print("Antwort wird über CABLE Input abgespielt ...")
    data, sr_in = sf.read(wav_out, dtype="float32", always_2d=True)
    data, sr_play = resample_if_needed(data, sr_in, OUT_SAMPLERATE)
    data = to_stereo(data)
    sd.play(data, sr_play, device=OUT_DEVICE_ID, blocking=True)
    print("Fertig.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FEHLER:", e)
