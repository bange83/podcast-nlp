import json
from pathlib import Path
from typing import Any

from .config import (
    AUDIO_DIR,
    DATA_DIR,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TRANSCRIBER_BACKEND,
    DEFAULT_WHISPER_MODEL,
    HF_TOKEN,
    REPORTS_DIR,
    TRANSCRIPTS_DIR,
)
from .downloader import PodcastDownloader
from .nlp_analyzer import LocalNLPAnalyzer
from .speaker_analyzer import SpeakerAnalyzer
from .transcriber import AudioTranscriber


class PodcastPipeline:
    def __init__(
        self,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        whisper_backend: str = DEFAULT_TRANSCRIBER_BACKEND,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        hf_token: str | None = HF_TOKEN,
        cpu_threads: int = 8,
    ):
        self.downloader = PodcastDownloader()
        token = hf_token or HF_TOKEN
        self.transcriber = AudioTranscriber(
            model_size=whisper_model,
            backend=whisper_backend,
            hf_token=token,
            cpu_threads=cpu_threads,
        )
        self.analyzer = LocalNLPAnalyzer(model_name=ollama_model)
        self.speaker_analyzer = SpeakerAnalyzer(model_name=ollama_model)

    # =========================================================================
    # Modulare Einzelschritte: Download -> Transkription -> NLP-Analyse
    # =========================================================================

    def download_episodes(
        self,
        query: str | None = None,
        feed_url: str | None = None,
        limit: int = 1,
    ) -> list[Path]:
        """Schritt 1: Lädt eine oder mehrere Folgen herunter und speichert Audio + Metadaten."""
        podcast_title = None
        podcast_author = None

        if query:
            print(f"\n🔍 Suche nach Podcast: '{query}'...")
            results = self.downloader.search_podcasts(query, limit=3)
            if not results:
                raise ValueError(f"Kein Podcast für die Suche '{query}' gefunden.")
            best_match = results[0]
            podcast_title = best_match.get("name")
            podcast_author = best_match.get("artist")
            feed_url = best_match["feed_url"]
            print(f"✅ Treffer gefunden: '{podcast_title}' von {podcast_author}")
            print(f"📡 Feed-URL: {feed_url}")
        if not feed_url:
            raise ValueError("Bitte entweder --search oder --feed angeben.")

        print(f"\n📡 Lese Feed ({limit} Folge{'n' if limit > 1 else ''}): {feed_url}")
        episodes = self.downloader.fetch_feed_episodes(feed_url, limit=limit)
        downloaded = []
        for i, ep in enumerate(episodes, 1):
            if not ep.podcast_title and podcast_title:
                ep.podcast_title = podcast_title
            if not ep.podcast_author and podcast_author:
                ep.podcast_author = podcast_author
            print(f"\n>>> [{i}/{len(episodes)}] Download: '{ep.title}'")
            path = self.downloader.download_episode(ep)
            downloaded.append(path)

        print(f"\n✅ Download abgeschlossen: {len(downloaded)} Datei(en) gespeichert.")
        return downloaded

    def transcribe_batch(
        self,
        audio_file: Path | None = None,
        language: str | None = None,
        force: bool = False,
    ) -> list[Path]:
        """Schritt 2: Transkribiert einzelne oder alle Audiodateien (Batch über data/<podcast>/audio/)."""
        if audio_file:
            if not audio_file.exists():
                raise FileNotFoundError(f"Audiodatei nicht gefunden: {audio_file}")
            targets = [audio_file]
        else:
            extensions = {".mp3", ".m4a", ".wav", ".aac", ".flac"}
            # Durchsuche alle data/<podcast>/audio/ sowie legacy data/audio/
            targets = sorted([
                f for f in DATA_DIR.rglob("*.*")
                if f.parent.name == "audio" and f.suffix.lower() in extensions
            ])
            if not targets:
                print(f"[Pipeline] Keine Audiodateien in {DATA_DIR} gefunden.")
                return []

        produced = []
        for i, path in enumerate(targets, 1):
            if path.parent.name == "audio":
                transcripts_dir = path.parent.parent / "transcripts"
            else:
                transcripts_dir = TRANSCRIPTS_DIR
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            json_target = transcripts_dir / f"{path.stem}.json"

            if not force and json_target.exists():
                print(f"[{i}/{len(targets)}] Transkript existiert bereits: {json_target.name}")
                produced.append(json_target)
                continue

            print(f"\n[{i}/{len(targets)}] 🎧 Transkribiere {path.name}...")
            self.transcriber.transcribe(
                path,
                language=language,
                force_recompute=force,
            )
            produced.append(json_target)

        print(f"\n✅ Transkription abgeschlossen: {len(produced)} Datei(en)")
        return produced

    def analyze_batch(
        self,
        transcript_file: Path | None = None,
        force: bool = False,
    ) -> list[Path]:
        """Schritt 3: Führt LLM-Analysen für Transkripte in data/<podcast>/transcripts/ durch (Batch)."""
        if transcript_file:
            if not transcript_file.exists():
                raise FileNotFoundError(f"Transkriptdatei nicht gefunden: {transcript_file}")
            # Falls eine .txt Datei angegeben wurde, bevorzugen wir die zugehörige .json
            if transcript_file.suffix == ".txt" and transcript_file.with_suffix(".json").exists():
                transcript_file = transcript_file.with_suffix(".json")
            targets = [transcript_file]
        else:
            targets = sorted([
                f for f in DATA_DIR.rglob("*.json")
                if f.parent.name == "transcripts" and not f.name.endswith(".meta.json")
            ])
            if not targets:
                print(f"[Pipeline] Keine Transkripte in {DATA_DIR} gefunden.")
                return []

        produced = []
        for i, path in enumerate(targets, 1):
            stem = path.stem
            if path.parent.name == "transcripts":
                reports_dir = path.parent.parent / "reports"
                audio_dir = path.parent.parent / "audio"
            else:
                reports_dir = REPORTS_DIR
                audio_dir = AUDIO_DIR
            reports_dir.mkdir(parents=True, exist_ok=True)

            report_json = reports_dir / f"{stem}_analysis.json"
            report_md = reports_dir / f"{stem}_report.md"

            if not force and report_json.exists() and report_md.exists():
                print(f"[{i}/{len(targets)}] Report existiert bereits: {report_md.name}")
                produced.append(report_json)
                continue

            print(f"\n[{i}/{len(targets)}] 🧠 Analysiere Transkript: {path.name}...")
            if path.suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    transcript_data = json.load(f)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                transcript_data = {
                    "audio_file": stem,
                    "full_text": raw_text,
                    "segments": [],
                }

            # Metadaten aus audio/<stem>.meta.json laden (falls vorhanden)
            meta_file = audio_dir / f"{stem}.meta.json"
            meta: dict[str, Any] = {}
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                except Exception:
                    pass

            ep_title = meta.get("title") or transcript_data.get("audio_file") or stem
            pod_title = meta.get("podcast_title")
            pod_author = meta.get("podcast_author")
            ep_desc = meta.get("description")

            # Sprecher-Namen dynamisch auflösen, falls noch nicht geschehen
            if transcript_data.get("has_diarization") and not transcript_data.get("speaker_mapping"):
                print("🗣️ Löse Sprecher-Namen dynamisch mit LLM auf...")
                transcript_data = self.analyzer.map_speaker_names(
                    transcript_data,
                    podcast_title=pod_title,
                    podcast_author=pod_author,
                    episode_title=ep_title,
                    episode_description=ep_desc,
                )
            self.analyzer.analyze_transcript(
                transcript_data,
                title=ep_title,
                output_dir=reports_dir,
                force_recompute_chapters=force,
            )

            # Aktualisiertes Transkript (.json und .txt) mit Kapiteln speichern
            with open(path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)
            txt_path = path.parent / f"{stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcript_data.get("full_text", ""))

            produced.append(report_json)

        print(f"\n✅ Analyse abgeschlossen: {len(produced)} Report(s)")
        return produced

    def analyze_speakers(
        self,
        transcript_file: Path | None = None,
        force: bool = False,
    ) -> list[Path]:
        """Führt quantitative und qualitative Sprecheranalysen auf Transkripten durch."""
        if transcript_file:
            if not transcript_file.exists():
                raise FileNotFoundError(f"Transkriptdatei nicht gefunden: {transcript_file}")
            if transcript_file.suffix == ".txt" and transcript_file.with_suffix(".json").exists():
                transcript_file = transcript_file.with_suffix(".json")
            targets = [transcript_file]
        else:
            targets = sorted([
                f for f in DATA_DIR.rglob("*.json")
                if f.parent.name == "transcripts" and not f.name.endswith(".meta.json")
            ])
            if not targets:
                print(f"[Pipeline] Keine Transkripte in {DATA_DIR} gefunden.")
                return []

        produced = []
        for i, path in enumerate(targets, 1):
            stem = path.stem
            if path.parent.name == "transcripts":
                reports_dir = path.parent.parent / "reports"
                audio_dir = path.parent.parent / "audio"
            else:
                reports_dir = REPORTS_DIR
                audio_dir = AUDIO_DIR
            reports_dir.mkdir(parents=True, exist_ok=True)

            report_md = reports_dir / f"{stem}_speakers.md"

            if not force and report_md.exists():
                print(f"[{i}/{len(targets)}] Sprecher-Analyse existiert bereits: {report_md.name}")
                produced.append(report_md)
                continue

            print(f"\n[{i}/{len(targets)}] 🎙️ Führe Sprecher-Analyse durch für: {path.name}...")
            with open(path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)

            meta_file = audio_dir / f"{stem}.meta.json"
            meta: dict[str, Any] = {}
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                except Exception:
                    pass

            ep_title = meta.get("title") or transcript_data.get("audio_file") or stem
            self.speaker_analyzer.run_analysis(transcript_data, title=ep_title, output_dir=reports_dir)
            produced.append(report_md)

        print(f"\n✅ Sprecher-Analyse abgeschlossen: {len(produced)} Report(s)")
        return produced

        return produced

    # =========================================================================
    # End-to-End Orchestrierung (Download + Transkription + Analyse in Folge)
    # =========================================================================

    def run_end_to_end(
        self,
        query: str | None = None,
        feed_url: str | None = None,
        file: Path | None = None,
        limit: int = 1,
        language: str | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Führt alle 3 Schritte in einem Rutsch aus."""
        if file:
            return [self.process_audio_file(file, language=language, force=force)]
        elif query:
            return self.process_search(query, limit=limit, language=language, force=force)
        elif feed_url:
            return self.process_feed(feed_url, limit=limit, language=language, force=force)
        else:
            raise ValueError("Weder --search, --feed noch --file angegeben.")

    def process_search(
        self,
        query: str,
        limit: int = 1,
        language: str | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Sucht nach einem Podcast und führt alle Schritte aus."""
        print(f"\n🔍 Suche nach Podcast: '{query}'...")
        results = self.downloader.search_podcasts(query, limit=3)
        if not results:
            raise ValueError(f"Kein Podcast für die Suche '{query}' gefunden.")

        best_match = results[0]
        print(f"✅ Treffer gefunden: '{best_match['name']}' von {best_match['artist']}")
        print(f"📡 Feed-URL: {best_match['feed_url']}")

        return self.process_feed(
            feed_url=best_match["feed_url"],
            limit=limit,
            language=language,
            force=force,
            podcast_title=best_match.get("name"),
            podcast_author=best_match.get("artist"),
        )

    def process_feed(
        self,
        feed_url: str,
        limit: int = 1,
        language: str | None = None,
        force: bool = False,
        podcast_title: str | None = None,
        podcast_author: str | None = None,
    ) -> list[dict[str, Any]]:
        """Verarbeitet Episoden aus einem RSS-Feed durch alle 3 Stufen."""
        print("\n==========================================")
        print(f"📡 1. Lese RSS-Feed: {feed_url}")
        print("==========================================")
        episodes = self.downloader.fetch_feed_episodes(feed_url, limit=limit)

        results = []
        for i, ep in enumerate(episodes, 1):
            if not ep.podcast_title and podcast_title:
                ep.podcast_title = podcast_title
            if not ep.podcast_author and podcast_author:
                ep.podcast_author = podcast_author

            print(f"\n>>> Verarbeite Episode {i}/{len(episodes)}: '{ep.title}'")
            # Stufe 1: Download
            audio_path = self.downloader.download_episode(ep)

            # Stufe 2: Transkription
            print("\n🎧 2. Transkription...")
            transcript_data = self.transcriber.transcribe(
                audio_path,
                language=language,
                force_recompute=force,
            )

            # Optionale Sprecher-Namensauflösung via LLM
            if transcript_data.get("has_diarization"):
                print("\n🗣️ Löse Sprecher-Namen dynamisch mit LLM auf...")
                transcript_data = self.analyzer.map_speaker_names(
                    transcript_data,
                    podcast_title=ep.podcast_title,
                    podcast_author=ep.podcast_author,
                    episode_title=ep.title,
                    episode_description=ep.description,
                )

            # Stufe 3: NLP Analyse
            print("\n🧠 3. NLP-Analyse mit lokalem LLM...")
            podcast_dir = audio_path.parent.parent if audio_path.parent.name == "audio" else None
            reports_dir = (podcast_dir / "reports") if podcast_dir else REPORTS_DIR
            reports_dir.mkdir(parents=True, exist_ok=True)
            analysis_data = self.analyzer.analyze_transcript(
                transcript_data,
                title=ep.title,
                output_dir=reports_dir,
            )

            results.append({
                "episode": ep.title,
                "audio_path": str(audio_path),
                "transcript": transcript_data,
                "analysis": analysis_data,
            })

        return results

    def process_audio_file(
        self,
        audio_path: Path,
        title: str | None = None,
        language: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Verarbeitet eine bereits lokal vorliegende Audiodatei (Stufe 2 & 3)."""
        print(f"\n🎧 2. Transkription von lokaler Datei: {audio_path.name}...")
        transcript_data = self.transcriber.transcribe(
            audio_path,
            language=language,
            force_recompute=force,
        )

        if transcript_data.get("has_diarization"):
            print("\n🗣️ Löse Sprecher-Namen dynamisch mit LLM auf...")
            transcript_data = self.analyzer.map_speaker_names(
                transcript_data,
                episode_title=title or audio_path.stem,
            )

        print("\n🧠 3. NLP-Analyse mit lokalem LLM...")
        podcast_dir = audio_path.parent.parent if audio_path.parent.name == "audio" else None
        reports_dir = (podcast_dir / "reports") if podcast_dir else REPORTS_DIR
        reports_dir.mkdir(parents=True, exist_ok=True)
        analysis_data = self.analyzer.analyze_transcript(
            transcript_data,
            title=title or audio_path.stem,
            output_dir=reports_dir,
        )

        return {
            "audio_path": str(audio_path),
            "transcript": transcript_data,
            "analysis": analysis_data,
        }
