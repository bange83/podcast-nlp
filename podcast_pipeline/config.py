import os
from pathlib import Path

from dotenv import load_dotenv

# Lädt Umgebungsvariablen aus .env (z.B. HF_TOKEN)
load_dotenv()

# Basis-Pfade
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
REPORTS_DIR = DATA_DIR / "reports"

# Standardmodelle
# Whisper: "tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"
DEFAULT_WHISPER_MODEL = "small"

# Whisper Compute Type auf macOS / Apple Silicon:
# "int8" nutzt ARM NEON Vektor-Beschleunigung (3-4x schneller auf M1-M4)
DEFAULT_COMPUTE_TYPE = "int8"

# Anzahl CPU-Threads für CTranslate2 (optimal für M-Serie Macs)
DEFAULT_CPU_THREADS = 8

# Standard Ollama Modell (passend zu Ihren installierten Modellen)
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "dolphin-llama3:8b")

# Hugging Face Token für PyAnnote Speaker Diarization (optional)
HF_TOKEN = os.getenv("HF_TOKEN")

