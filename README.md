# Podcast NLP Pipeline 🎙️🧠

Eine rein lokale Python-Pipeline zum Herunterladen von Podcasts, Transkribieren mit Whisper und Durchführen strukturierter NLP-Analysen mit lokalen Large Language Models (LLMs) via Ollama.

## Highlights (v0.2.0)

- **100% Lokal & Privat:** Keine Audiodaten oder Transkripte verlassen Ihren Rechner.
- **Hardwarebeschleunigung (Apple Silicon GPU & CPU/CUDA):**
  - **Apple Silicon Metal GPU:** Native Unterstützung für `mlx-whisper` auf Apple Silicon Macs (M1–M4) für extrem schnelle Transkriptionen (30–40x Echtzeit).
  - **Multi-Plattform / CPU & CUDA:** Nahtloser Fallback auf `faster-whisper` (CTranslate2 mit ARM NEON int8 oder CUDA) auf Linux, Windows und Intel-Rechnern.
  - **Intelligente Auto-Erkennung:** `--backend auto` wählt automatisch die schnellste Engine für Ihr System.
- **Schritt 1: Download:** Automatischer Abruf von Episoden aus RSS-Feeds oder per Apple Podcasts Suche inklusive Fortschrittsanzeige und Metadaten-Cache.
- **Schritt 2: Transkription & Sprechererkennung:** Zeitstempel-Segmentierung und automatische Sprecher-Diarization (Sprechertrennung Jan vs. Olli) via `pyannote.audio` (mit Apple Silicon MPS Beschleunigung).
- **Schritt 3: Lokale NLP-Analyse:** Themen- und Kapitel-Segmentierung mit Timestamps, Key Takeaways, Zusammenfassungen, Named Entity Recognition (NER) und dynamische Sprechernamen-Auflösung via [Ollama](https://ollama.com/) (z.B. `qwen2.5:14b` oder `dolphin-llama3:8b`).
- **Schritt 4: Sprecher- & Profiling-Analyse (Optional):** Quantitative Sprecherstatistiken (Redezeit, Wortanzahl, Sprechgeschwindigkeit WPM, längste Monologe, Top-Füllwörter) sowie psychologisches Charakter-, Werte- und Beziehungs-Profiling.
- **Reports:** Ausgabe sowohl als maschinenlesbares JSON als auch als lesbarer Markdown-Report.

---

## Projektstruktur

```text
podcast-nlp/
├── data/
│   ├── audio/          # Heruntergeladene Episoden (.mp3 / .m4a)
│   ├── transcripts/    # Transkripte (.json mit Timestamps & .txt)
│   └── reports/        # Analyse-Reports (.json & .md)
├── podcast_pipeline/
│   ├── __init__.py
│   ├── config.py          # Modell- & Pfadkonfiguration
│   ├── downloader.py      # RSS-Feed & Audio-Download
│   ├── transcriber.py     # Multi-Backend Transkription (mlx-whisper / faster-whisper)
│   ├── nlp_analyzer.py    # Ollama LLM-Analyse & Themenextraktion
│   ├── speaker_analyzer.py# Sprecherstatistiken & Charakterprofiling
│   └── pipeline.py        # Orchestrierung der Pipeline-Schritte
├── main.py             # CLI-Interface
├── requirements.txt
└── README.md
```

---

## Voraussetzungen

- **Python 3.10+**
- **FFmpeg** (für die Audioverarbeitung):

  ```bash
  brew install ffmpeg
  ```

- **Ollama** (für lokale LLMs):

  ```bash
  brew install ollama
  # Gewünschtes Modell herunterladen:
  ollama pull llama3.2
  ```

---

## Sprechererkennung einrichten (Optional, für Jan vs. Olli)

Für die Sprechererkennung (`pyannote.audio`) wird ein kostenloser Hugging Face Account benötigt:

1. Registrieren bei [huggingface.co](https://huggingface.co/)
2. Nutzungsbedingungen dieser beiden Modelle mit einem Klick akzeptieren:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Access Token generieren unter [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Read-Rechte).
4. Eine Datei `.env` im Projektordner anlegen (Vorlage `.env.example`):

   ```bash
   cp .env.example .env
   ```

   und dort `HF_TOKEN=hf_...` eintragen.

*Hinweis: Wenn kein Token eingetragen ist, läuft die Transkription ganz normal ohne Sprechertrennung weiter.*

---

## Installation & Ausführung mit uv (Empfohlen)

Mit [`uv`](https://docs.astral.sh/uv/) geht die Einrichtung blitzschnell und ohne Abhängigkeitskonflikte (nutzt automatisch die passende Python 3.11/3.12 Version mit vorkompilierten PyTorch/Whisper-Wheels):

```bash
cd /Volumes/Samsung990/git/podcast-nlp

# 1. Virtuelle Umgebung anlegen & Abhängigkeiten synchronisieren
uv sync

# 2. Pipeline direkt über uv ausführen:
uv run main.py --search "Fest & Flauschig" --limit 1
```

Alternativ klassisch mit venv:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python3 main.py --search "Fest & Flauschig" --limit 1
```

---

## Benutzung

Das Skript ist modular aufgebaut: Sie können die drei Phasen (Download, Transkription, NLP-Analyse) komplett getrennt voneinander oder in einem Rutsch ausführen.

### 1. Schritt: Episoden herunterladen (`download`)
Lädt z. B. mehrere Episoden im Hintergrund herunter und speichert Audio-Dateien sowie deren Metadaten in `data/audio/`:

```bash
# Nach Podcast-Namen suchen und die neuesten 3 Episoden herunterladen:
python3 main.py download --search "Fest & Flauschig" --limit 3

# Oder direkt per RSS-Feed:
python3 main.py download --feed "https://feeds.megaphone.fm/GLT4060321204" --limit 5
```

### 2. Schritt: Transkription & Sprechererkennung (`transcribe`)
Verarbeitet alle heruntergeladenen Audio-Dateien in `data/audio/`, die noch kein Transkript haben (Batch-Modus):

```bash
# Alle noch nicht transkribierten Episoden im Batch abarbeiten (Standard: --backend auto):
python3 main.py transcribe

# Explizit das Apple Silicon GPU Metal-Backend anfordern:
python3 main.py transcribe --backend mlx

# Oder auf CPU/CUDA mit CTranslate2 (faster-whisper) zwingen:
python3 main.py transcribe --backend faster-whisper

# Gezielt eine einzelne Datei transkribieren:
python3 main.py transcribe --file data/audio/meine_aufnahme.mp3

# Mit angepasster Modellgröße oder Threads:
python3 main.py transcribe --whisper-model large-v3-turbo --threads 8
```

### 3. Schritt: LLM-Analyse & Reports (`analyze`)
Liest die Transkripte aus `data/transcripts/`, ordnet Sprechernamen zu und erzeugt strukturierte Markdown- & JSON-Reports in `data/reports/`:

```bash
# Alle Transkripte analysieren, die noch keinen Report haben:
python3 main.py analyze

# Mit einem anderen lokalen Ollama-Modell:
python3 main.py analyze --model llama3.2

# Nur eine bestimmte Transkript-Datei analysieren:
python3 main.py analyze --file data/transcripts/meine_episode.json
```

### 4. Optional: Detaillierte Sprecher-Analyse (`speakers`)
Erstellt für fertige Transkripte eine tiefgehende quantitative Statistik (Redeanteile, Wortzahlen, Sprechtempo in WPM, längste Monologe, Top-Füllwörter) sowie ein psychologisches Charakter-, Werte- und Beziehungs-Profiling via LLM:

```bash
# Alle vorhandenen Transkripte analysieren:
python3 main.py speakers

# Mit spezifischem Modell (z.B. Qwen 2.5 14B):
python3 main.py speakers --model qwen2.5:14b
```

---

### Alternative: Alles in einem Rutsch (`run`)
Falls Sie eine Episode sofort von A bis Z durchlaufen lassen möchten:

```bash
python3 main.py run --search "Fest & Flauschig" --limit 1
```

---

## Lizenz

MIT
