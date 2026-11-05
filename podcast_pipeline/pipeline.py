from pathlib import Path
from typing import Optional, List, Dict, Any
from .config import DEFAULT_WHISPER_MODEL, DEFAULT_OLLAMA_MODEL
from .downloader import PodcastDownloader, PodcastEpisode
from .transcriber import AudioTranscriber
from .nlp_analyzer import LocalNLPAnalyzer


class PodcastPipeline:
    def __init__(
        self,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ):
        self.downloader = PodcastDownloader()
        self.transcriber = AudioTranscriber(model_size=whisper_model)
        self.analyzer = LocalNLPAnalyzer(model_name=ollama_model)

    def process_search(
        self,
        query: str,
        limit: int = 1,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Sucht nach einem Podcast-Namen, wählt den ersten Treffer und verarbeitet ihn."""
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
            language=language
        )

    def process_feed(
        self,
        feed_url: str,
        limit: int = 1,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Verarbeitet Episoden aus einem RSS-Feed durch alle 3 Stufen."""
        print(f"\n==========================================")
        print(f"📡 1. Lese RSS-Feed: {feed_url}")
        print(f"==========================================")
        episodes = self.downloader.fetch_feed_episodes(feed_url, limit=limit)

        results = []
        for i, ep in enumerate(episodes, 1):
            print(f"\n>>> Verarbeite Episode {i}/{len(episodes)}: '{ep.title}'")
            # Stufe 1: Download
            audio_path = self.downloader.download_episode(ep)

            # Stufe 2: Transkription
            print(f"\n🎧 2. Transkription...")
            transcript_data = self.transcriber.transcribe(audio_path, language=language)

            # Stufe 3: NLP Analyse
            print(f"\n🧠 3. NLP-Analyse mit lokalem LLM...")
            analysis_data = self.analyzer.analyze_transcript(transcript_data, title=ep.title)

            results.append({
                "episode": ep.title,
                "audio_path": str(audio_path),
                "transcript": transcript_data,
                "analysis": analysis_data
            })

        return results

    def process_audio_file(
        self,
        audio_path: Path,
        title: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verarbeitet eine bereits lokal vorliegende Audiodatei (Stufe 2 & 3)."""
        print(f"\n🎧 2. Transkription von lokaler Datei: {audio_path.name}...")
        transcript_data = self.transcriber.transcribe(audio_path, language=language)

        print(f"\n🧠 3. NLP-Analyse mit lokalem LLM...")
        analysis_data = self.analyzer.analyze_transcript(transcript_data, title=title or audio_path.stem)

        return {
            "audio_path": str(audio_path),
            "transcript": transcript_data,
            "analysis": analysis_data
        }
