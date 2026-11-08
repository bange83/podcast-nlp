# Podcast NLP Pipeline 🎙️🧠

Eine rein lokale, hardwarebeschleunigte Python-Pipeline zum Herunterladen von Podcasts, Transkribieren mit Whisper und Durchführen tiefgehender NLP- und Konversationsanalysen mit lokalen Large Language Models (LLMs) via Ollama.

## Highlights (v0.3.0)

- **100% Lokal & Privat:** Keine Audiodaten oder Transkripte verlassen Ihren Rechner.
- **Hardwarebeschleunigung (Apple Silicon GPU & CPU/CUDA):**
  - **Apple Silicon Metal GPU:** Native Unterstützung für `mlx-whisper` auf Apple Silicon Macs (M1–M4) für extrem schnelle Transkriptionen (30–40x Echtzeit).
  - **Multi-Plattform / CPU & CUDA:** Nahtloser Fallback auf `faster-whisper` (CTranslate2 mit ARM NEON int8 oder CUDA) auf Linux, Windows und Intel-Rechnern.
  - **Intelligente Auto-Erkennung:** `--backend auto` wählt automatisch die schnellste Engine für Ihr System.
- **Schritt 1: Download & Nummerierung:** Automatischer Abruf von Episoden aus RSS-Feeds oder per Apple Podcasts Suche. Saubere Strukturierung nach Podcast-Namen und 4-stelliger Episodennummer (z. B. `data/lanz-precht/audio/0270_...mp3`).
- **Schritt 2: Transkription & Akustische Sprechertrennung:** Robuste Sprecher-Diarization via `pyannote.audio` (mit Apple Silicon MPS-Beschleunigung und automatischer 16 kHz Mono-WAV-Pufferung für fehlerfreie VBR-MP3-Verarbeitung).
- **Schritt 3: Lückenlose Kapitel & NLP-Analyse:**
  - Themengetriebene Kapitelblöcke mit Start- und Endzeitstempeln, Kurzbeschreibung, konkreten Stichpunkten (`bullet_points`) und Schlagworten.
  - Vollständige Episodenabdeckung von `00:00` bis zum Schluss (keine künstlichen Kapitelquoten oder Text-Cutoffs).
  - TL;DR, detaillierte Zusammenfassung, Key Takeaways und Named Entity Recognition (NER) via [Ollama](https://ollama.com/) (empfohlen: `qwen2.5:14b`).
- **Schritt 4: Erweiterte Konversations- & Sprecheranalyse (v0.3.0):**
  - **Quantitative Sprecher-Metriken:** Exakte Redezeit, Wortanzahl, Sprechtempo (WPM), Fragemuster-Quote und längste Monologe.
  - **Echte akustische Unterbrechungen:** Unterscheidung zwischen ausgeteilten und erlittenen Overlaps direkt aus den PyAnnote-RTTM-Daten.
  - **Qualitativer Unterbrechungs-Stil (Micro-Batching):** KI-Klassifikation in *Steering* (Themenlenkung/Nachhaken), *Supportive* (Kooperativ/Zustimmung/Gag vollenden) und *Disruptive* (Gegenargument/Abwürgen).
  - **Zuhör- & Resonanz-Score (Listening Score):** Misst auf einer Skala von 1.0 bis 5.0, wie tief ein Sprecher bei der Übernahme auf das Argument des Vorredners eingeht.
  - **Duale Wortschatz-Vielfalt (MTLD):** Sowohl *Inhaltswort-MTLD* (reine thematische Fachlexik) als auch *Gesamt-MTLD* (Alltagssprache inklusive rhetorischer Schleifen).
  - **Gefilterte Themen-Substantive:** Konversationsfloskeln werden entfernt, sodass thematische Kernbegriffe (*Dollar, Trump, Kapitalismus, Film*) hervortreten.
  - **Psychologisches Profiling:** Archetypen, Rollen, Tonalität, Weltanschauung und Beziehungsdynamik zwischen den Moderatoren.
- **Reports:** Ausgabe sowohl als maschinenlesbares JSON (`_analysis.json`, `_speakers.json`) als auch als lesbare Markdown-Reports (`_report.md`, `_speakers.md`).

---

## 📊 Showcases & Beispiel-Reports

Vollständige, generierte Beispiel-Auswertungen aus realen Podcast-Episoden sind im Ordner [`examples/`](./examples) abgelegt:

- **Lanz & Precht (Ausgabe #270):**
  - [🎙️ Sprecher- & Konversations-Report](./examples/lanz-precht_0270_speakers.md) *(Quantitativ, Unterbrechungsstile, Zuhör-Score & psychologische Profile)*
  - [⏱️ Themen- & Kapitel-Report](./examples/lanz-precht_0270_chapters.md) *(Lückenlose Zeitstempel, TL;DR, Bullet Points & Entitäten)*
- **Fest & Flauschig (Auf einen Blunt mit Selenskyj):**
  - [🎙️ Sprecher- & Konversations-Report](./examples/fest-flauschig_0719_speakers.md) *(Jan Böhmermann vs. Olli Schulz: 247 WPM, 78% disruptive Anekdoten)*
- **Kack & Sachgeschichten (Folge #368):**
  - [🎙️ Sprecher- & Konversations-Report](./examples/kack-sachgeschichten_0368_speakers.md) *(Freddy vs. Tobi: Ping-Pong-Dynamik, 58% supportive interruptions)*

### Preview: Was die Sprecheranalyse liefert (Lanz vs. Precht)

| Metrik | **Markus Lanz** | **Richard David Precht** | Was die KI daraus liest |
|---|---|---|---|
| **Sprechtempo** | **166.8** WPM | **183.6** WPM | Precht formuliert schneller; Lanz setzt gezieltere Pausen. |
| **Zuhör- & Resonanz-Score** | **4.1** / 5.0 | **3.0** / 5.0 | Lanz greift Prechts Thesen direkt auf; Precht wechselt häufiger auf Meta-Ebene. |
| **Unterbrechungs-Stil** | 33% Steer / **67% Supp** | **60% Steer** / 27% Supp / 13% Disr | Lanz bestärkt & hakt nach; Precht lenkt aktiv den Diskurs. |
| **Wortschatz-Vielfalt (MTLD Inhalt)** | **418.3** | **456.9** | Prechts Wortschatz bei echten Inhaltswörtern (thematische Tiefe) ist messbar breiter. |
| **Wortschatz-Vielfalt (MTLD Gesamt)** | **118.6** | **88.7** | *MTLD-Paradox:* Im Gesamtwortschatz wiederholt Precht Denkfloskeln (*"ja"*, *"das ist das Ding"*). |
| **Top-Themenwörter** | `Dollar`, `Trump`, `Milliarden` | `Trump`, `Kapitalismus`, `Bevölkerung` | Lanz quantifiziert das Geld; Precht analysiert das gesellschaftliche System. |

---

## Projektstruktur

```text
podcast-nlp/
├── data/
│   ├── <podcast-slug>/     # z. B. lanz-precht/, fest-flauschig/, kack-sachgeschichten/
│   │   ├── audio/          # Audio-Dateien (z. B. 0270_titel.mp3) & .meta.json
│   │   ├── transcripts/    # Transkripte (.json mit Timestamps, .txt & .rttm Diarization)
│   │   └── reports/        # Themen- & Sprecher-Reports (.json & .md)
├── podcast_pipeline/
│   ├── __init__.py
│   ├── config.py           # Modell- & Pfadkonfiguration
│   ├── downloader.py       # RSS-Feed & Audio-Download
│   ├── transcriber.py      # Multi-Backend Transkription (mlx-whisper / faster-whisper & PyAnnote)
│   ├── nlp_analyzer.py     # Flexible Themen- & Kapitelanalyse via Ollama
│   ├── speaker_analyzer.py # Konversationsanalyse, MTLD, Interaktionsstile & Profiling
│   └── pipeline.py         # End-to-End Orchestrierung & Batching
├── main.py                 # CLI-Interface
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Voraussetzungen

- **Python 3.11 oder 3.12**
- **FFmpeg** (für die Audioverarbeitung):

  ```bash
  brew install ffmpeg
  ```

- **Ollama** (für lokale LLMs):

  ```bash
  brew install ollama
  # Empfohlenes deutsches Modell herunterladen:
  ollama pull qwen2.5:14b
  ```

---

## Sprechererkennung einrichten (Optional, für Diarization)

Für die Sprechertrennung (`pyannote.audio`) wird ein kostenloser Hugging Face Account benötigt:

1. Registrieren bei [huggingface.co](https://huggingface.co/)
2. Nutzungsbedingungen dieser beiden Modelle mit einem Klick akzeptieren:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Access Token generieren unter [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Read-Rechte).
4. Eine Datei `.env` im Projektordner anlegen:

   ```bash
   cp .env.example .env
   ```

   und dort `HF_TOKEN=hf_...` eintragen.

*Hinweis: Wenn kein Token eingetragen ist, läuft die Transkription ganz normal ohne Sprechertrennung weiter.*

---

## Installation & Ausführung

Mit [`uv`](https://docs.astral.sh/uv/) (empfohlen für maximale Geschwindigkeit):

```bash
git clone https://github.com/bange83/podcast-nlp.git
cd podcast-nlp

# 1. Virtuelle Umgebung anlegen & Abhängigkeiten synchronisieren
uv sync

# 2. Pipeline direkt über uv ausführen:
uv run python main.py download --search "Lanz & Precht" --limit 1
```

Alternativ klassisch mit venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py download --search "Lanz & Precht" --limit 1
```

---

## Benutzung

Das CLI ist modular aufgebaut: Sie können die Phasen komplett getrennt voneinander oder in einem Rutsch ausführen.

### 1. Schritt: Episoden herunterladen (`download`)
Lädt Episoden herunter und legt sie strukturiert unter `data/<podcast>/audio/<NNNN>_<slug>.mp3` ab:

```bash
# Nach Podcast-Namen suchen und die neueste Folge herunterladen:
python main.py download --search "Lanz & Precht" --limit 1

# Oder nach anderen Podcasts suchen:
python main.py download --search "Fest & Flauschig" --limit 1
python main.py download --search "Kack & Sachgeschichten" --limit 1

# Oder direkt per RSS-Feed:
python main.py download --feed "https://feeds.megaphone.fm/GLT4060321204" --limit 3
```

### 2. Schritt: Transkription & Diarization (`transcribe`)
Verarbeitet alle Audiodateien, die noch kein Transkript haben (oder holt fehlende Sprecher-Diarization nach):

```bash
# Batch-Modus über alle Podcasts (Standard: --backend auto):
python main.py transcribe

# Explizit Metal GPU Beschleunigung auf Apple Silicon anfordern:
python main.py transcribe --backend mlx

# Gezielt eine einzelne Datei transkribieren:
python main.py transcribe --file data/lanz-precht/audio/0270_...mp3
```

### 3. Schritt: Themen- & Kapitel-Analyse (`analyze`)
Generiert die lückenlose Kapitelübersicht samt Zusammenfassungen, Key Takeaways und Entitäten:

```bash
# Alle Transkripte im Batch analysieren (Standard: qwen2.5:14b):
python main.py analyze --model qwen2.5:14b

# Einzelnes Transkript neu generieren:
python main.py analyze --file data/lanz-precht/transcripts/0270_...json --force
```

### 4. Schritt: Erweiterte Sprecher- & Konversations-Analyse (`speakers`)
Erstellt die quantitative Sprecher-Statistik (WPM, Fragemuster, Unterbrechungen, MTLD, Top-Themenwörter), klassifiziert Interaktionsstile via Micro-Batching und erstellt das psychologische Profil:

```bash
# Alle Sprecher-Reports im Batch generieren:
python main.py speakers --model qwen2.5:14b

# Einzelne Folge analysieren oder mit --force neu berechnen:
python main.py speakers --file data/lanz-precht/transcripts/0270_...json --force
```

---

### End-to-End Orchestrierung (`run`)
Führt alle 4 Schritte (Download, Transkription & Diarization, NLP-Kapitelanalyse, Sprecher-Profiling) in einem einzigen Aufruf aus:

```bash
# Mit uv (empfohlen):
uv run python main.py run --search "Lanz & Precht" --limit 1

# Klassisch mit aktiviertem venv:
python main.py run --search "Lanz & Precht" --limit 1

# Anderen Podcast suchen oder explizites Modell angeben (Standard: qwen2.5:14b):
uv run python main.py run --search "Krewkast" --limit 1 --model qwen2.5:14b

# Lokale Audiodatei direkt komplett verarbeiten:
uv run python main.py run --file data/lanz-precht/audio/0270_...mp3
```

---

## Lizenz

MIT
