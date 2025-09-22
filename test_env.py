import os
from dotenv import load_dotenv

load_dotenv()  # liest .env ein

print("OpenAI:", os.getenv("OPENAI_API_KEY"))
print("ElevenLabs:", os.getenv("ELEVENLABS_API_KEY"))
