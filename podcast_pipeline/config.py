from pathlib import Path

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
# "float16", "int8", "float32"
DEFAULT_COMPUTE_TYPE = "float32"

# Ollama Modell für NLP-Analysen (z.B. "llama3.2", "mistral", "qwen2.5:7b")
DEFAULT_OLLAMA_MODEL = "llama3.2"
