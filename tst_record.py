import sounddevice as sd
from scipy.io.wavfile import write

TARGET_NAME = "CABLE Output"  # nimmt den ersten Eingang, der so heißt
CHANNELS = 1                  # Mono reicht

# --- Device suchen ---
devices = sd.query_devices()
device_id = None
for i, d in enumerate(devices):
    name = d["name"]
    if TARGET_NAME.lower() in name.lower() and d["max_input_channels"] > 0:
        device_id = i
        default_sr = int(d.get("default_samplerate") or 48000)
        print(f"Gefunden: [{i}] {name} | default SR={default_sr}")
        break

if device_id is None:
    raise RuntimeError(f'Kein Input-Device mit "{TARGET_NAME}" gefunden.')

# --- Samplerate festlegen (Fallback, falls 48k nicht geht, nimm 44100) ---
samplerate = default_sr if default_sr in (48000, 44100) else 48000

print(f"Starte 3s Aufnahme @ {samplerate} Hz von Device [{device_id}] ...")
rec = sd.rec(
    int(3 * samplerate),
    samplerate=samplerate,
    channels=CHANNELS,
    dtype="int16",
    device=device_id
)
sd.wait()

out_wav = "cable_test.wav"
write(out_wav, samplerate, rec)
print(f"Fertig! Gespeichert: {out_wav}")
