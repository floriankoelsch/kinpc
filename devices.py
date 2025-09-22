import sounddevice as sd

print("==== Verfügbare Audio-Geräte ====")
devices = sd.query_devices()
for idx, dev in enumerate(devices):
    print(f"[{idx}] {dev['name']} ({dev['max_input_channels']} In / {dev['max_output_channels']} Out)")
