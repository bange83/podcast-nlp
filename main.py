#!/usr/bin/env python3
import argparse
from pathlib import Path
from podcast_pipeline.pipeline import PodcastPipeline
from podcast_pipeline.config import DEFAULT_WHISPER_MODEL, DEFAULT_OLLAMA_MODEL


def main():
    parser = argparse.ArgumentParser(
        description="Podcast NLP Pipeline: Automatischer Download, lokale Whisper-Transkription & LLM-Analyse mit Ollama."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--feed", "-f",
        type=str,
        help="URL des Podcast-RSS-Feeds"
    )
    group.add_argument(
        "--search", "-s",
        type=str,
        help="Name des Podcasts für automatische Suche (z.B. 'Fest & Flauschig')"
    )
    group.add_argument(
        "--file", "-i",
        type=str,
        help="Pfad zu einer lokalen Audiodatei (.mp3, .m4a, .wav)"
    )

    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=1,
        help="Anzahl der neuesten Episoden aus dem RSS-Feed, die verarbeitet werden sollen (Standard: 1)"
    )
    parser.add_argument(
        "--whisper-model", "-w",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        help=f"Whisper Modellgröße: tiny, base, small, medium, large-v3-turbo (Standard: {DEFAULT_WHISPER_MODEL})"
    )
    parser.add_argument(
        "--llm-model", "-m",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama Modellname: z.B. llama3.2, mistral, qwen2.5 (Standard: {DEFAULT_OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Sprachcode für Whisper erzwingen (z.B. 'de' oder 'en'). Wenn weggelassen, automatische Erkennung."
    )

    args = parser.parse_args()

    pipeline = PodcastPipeline(
        whisper_model=args.whisper_model,
        ollama_model=args.llm_model
    )

    if args.feed:
        pipeline.process_feed(
            feed_url=args.feed,
            limit=args.limit,
            language=args.language
        )
    elif args.search:
        pipeline.process_search(
            query=args.search,
            limit=args.limit,
            language=args.language
        )
    elif args.file:
        file_path = Path(args.file).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
        pipeline.process_audio_file(
            audio_path=file_path,
            language=args.language
        )

    print("\n✨ Pipeline erfolgreich durchgelaufen! Ergebnisse finden sich im Ordner 'data/'.")


if __name__ == "__main__":
    main()
