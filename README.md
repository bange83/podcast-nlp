# Podcast NLP Pipeline 🎙️🧠

Eine rein lokale Python-Pipeline zum Herunterladen von Podcasts, Transkribieren mit Whisper und Durchführen strukturierter NLP-Analysen mit lokalen Large Language Models (LLMs) via Ollama.

## Highlights
- **100% Lokal & Privat:** Keine Audiodaten oder Transkripte verlassen Ihren Rechner.
- **Schritt 1: Download:** Automatischer Abruf der neuesten Episoden aus RSS-Feeds inklusive Fortschrittsanzeige.
- **Schritt 2: Transkription:** Lokales Speech-to-Text mittels `faster-whisper` (CTranslate2) inklusive Zeitstempel-Segmentierung und VAD (Voice Activity Detection).
- **Schritt 3: Lokale NLP-Analyse:** Extrahieren von Zusammenfassungen, Key Takeaways, Themen, Entitäten (NER) und Tonalität/Sentiment über lokale Modelle in [Ollama](https://ollama.com/) (z.B. `llama3.2`, `qwen2.5`, `mistral`).
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

## Installation

1. Virtuelle Umgebung erstellen und aktivieren:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

---

## Benutzung

### 1. Podcast per Name suchen & analysieren
Einfach nach dem Namen suchen (nutzt das offene Apple-Podcasts-Verzeichnis):
```bash
python3 main.py --search "Fest & Flauschig" --limit 1
```

### 2. Podcast direkt per RSS-Feed analysieren
Die neueste Episode eines Feeds verarbeiten:
```bash
python3 main.py --feed "https://feeds.megaphone.fm/GLT4060321204" --limit 1
```

Die letzten 3 Episoden verarbeiten:
```bash
python3 main.py --feed "https://feeds.megaphone.fm/GLT4060321204" --limit 3
```

### 3. Eine bestehende lokale Audiodatei analysieren
```bash
python3 main.py --file "data/audio/meine_aufnahme.mp3"
```

### 3. Modelle anpassen
Sie können die Modellgröße für Whisper und das LLM in Ollama direkt über Flags steuern:
```bash
python3 main.py \
  --feed "https://example.com/podcast/feed.xml" \
  --whisper-model small \
  --llm-model qwen2.5:7b \
  --language de
```

---

## Lizenz
MIT
