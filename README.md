# Podcast NLP Pipeline 🎙️🧠

Eine rein lokale Python-Pipeline zum Herunterladen von Podcasts, Transkribieren mit Whisper und Durchführen strukturierter NLP-Analysen mit lokalen Large Language Models (LLMs) via Ollama.

## Highlights

- **100% Lokal & Privat:** Keine Audiodaten oder Transkripte verlassen Ihren Rechner.
- **Schritt 1: Download:** Automatischer Abruf der neuesten Episoden aus RSS-Feeds inklusive Fortschrittsanzeige.
- **Schritt 2: Transkription & Sprechererkennung:** Lokales Speech-to-Text mittels `faster-whisper` (CTranslate2) inklusive Zeitstempel-Segmentierung und optionaler Sprecher-Diarization (Sprechertrennung Jan vs. Olli) via `pyannote.audio`.
- **Schritt 3: Lokale NLP-Analyse:** Extrahieren von Zusammenfassungen, Key Takeaways, Themen, Entitäten (NER) und Tonalität/Sentiment über lokale Modelle in [Ollama](https://ollama.com/) (Standard: `dolphin-llama3:8b`). Erkennt automatisch die echten Namen der Sprecher aus dem Dialog.
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
│   ├── config.py       # Modell- & Pfadkonfiguration
│   ├── downloader.py   # RSS-Feed & Audio-Download
│   ├── transcriber.py  # faster-whisper Transkription
│   ├── nlp_analyzer.py # Ollama LLM-Analyse
│   └── pipeline.py     # End-to-End Orchestrierung
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
# Alle noch nicht transkribierten Episoden im Batch abarbeiten:
python3 main.py transcribe

# Oder gezielt eine einzelne Datei:
python3 main.py transcribe --file data/audio/meine_aufnahme.mp3

# Mit angepasster Modellgröße oder Threads:
python3 main.py transcribe --whisper-model medium --threads 8
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

---

### Alternative: Alles in einem Rutsch (`run`)
Falls Sie eine Episode sofort von A bis Z durchlaufen lassen möchten:

```bash
python3 main.py run --search "Fest & Flauschig" --limit 1
```

---

## Lizenz

MIT
