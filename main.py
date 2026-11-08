#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from podcast_pipeline.config import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_WHISPER_MODEL,
    HF_TOKEN,
)
from podcast_pipeline.pipeline import PodcastPipeline


def add_common_transcribe_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--whisper-model", "-w",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        help=f"Whisper-Modellgröße: tiny, base, small, medium, large-v3-turbo (Standard: {DEFAULT_WHISPER_MODEL})",
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Sprachcode für Whisper (z.B. 'de' oder 'en'). Standard: Automatische Erkennung.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=HF_TOKEN,
        help="Hugging Face Access Token für PyAnnote Sprechererkennung (Standard: aus .env)",
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=8,
        help="Anzahl CPU-Threads für Whisper (Standard: 8, optimal für Apple Silicon M-Serie)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Podcast NLP Pipeline: Modulare Werkzeuge zum Herunterladen, Transkribieren und LLM-Analysieren von Podcasts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  1. Nur Download (z.B. 3 Folgen):
     python3 main.py download --search "Fest & Flauschig" --limit 3

  2. Nur Transkription (alle noch nicht transkribierten Audiodateien im Batch):
     python3 main.py transcribe

  3. Nur LLM-Analyse (alle Transkripte analysieren & Reports erstellen):
     python3 main.py analyze

  4. Alles in einem Rutsch (End-to-End):
     python3 main.py run --search "Fest & Flauschig" --limit 1
        """,
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Verfügbare Befehle")

    # -------------------------------------------------------------------------
    # Subcommand: download
    # -------------------------------------------------------------------------
    p_dl = subparsers.add_parser("download", help="Schritt 1: Folgen eines Podcasts herunterladen (in data/audio/)")
    g_dl = p_dl.add_mutually_exclusive_group(required=True)
    g_dl.add_argument("--search", "-s", type=str, help="Podcast-Name für offene Apple Podcasts Suche")
    g_dl.add_argument("--feed", "-f", type=str, help="Direkte URL zum RSS-Feed")
    p_dl.add_argument("--limit", "-n", type=int, default=1, help="Anzahl der Episoden (Standard: 1)")

    # -------------------------------------------------------------------------
    # Subcommand: transcribe
    # -------------------------------------------------------------------------
    p_tr = subparsers.add_parser("transcribe", help="Schritt 2: Audiodateien transkribieren & Sprecher trennen (in data/transcripts/)")
    p_tr.add_argument("--file", "-i", type=str, default=None, help="Einzelne Audiodatei (Standard: alle neuen in data/audio/)")
    p_tr.add_argument("--force", action="store_true", help="Bereits existierende Transkripte neu berechnen")
    add_common_transcribe_args(p_tr)

    # -------------------------------------------------------------------------
    # Subcommand: analyze
    # -------------------------------------------------------------------------
    p_an = subparsers.add_parser("analyze", help="Schritt 3: Lokale LLM-Analyse für Transkripte durchführen (in data/reports/)")
    p_an.add_argument("--file", "-i", type=str, default=None, help="Einzelnes Transkript (.json oder .txt, Standard: alle neuen in data/transcripts/)")
    p_an.add_argument("--model", "-m", type=str, default=DEFAULT_OLLAMA_MODEL, help=f"Ollama Modellname (Standard: {DEFAULT_OLLAMA_MODEL})")
    p_an.add_argument("--force", action="store_true", help="Bereits erstellte Reports neu generieren")

    # -------------------------------------------------------------------------
    # Subcommand: speakers (Optional)
    # -------------------------------------------------------------------------
    p_spk = subparsers.add_parser("speakers", help="Optional: Detaillierte Sprecher-Statistik (Redezeit, WPM, Top-Wörter) & LLM-Charakterprofiling")
    p_spk.add_argument("--file", "-i", type=str, default=None, help="Einzelnes Transkript (.json oder .txt, Standard: alle in data/transcripts/)")
    p_spk.add_argument("--model", "-m", type=str, default=DEFAULT_OLLAMA_MODEL, help=f"Ollama Modellname (Standard: {DEFAULT_OLLAMA_MODEL})")
    p_spk.add_argument("--force", action="store_true", help="Bereits erstellte Sprecher-Analysen neu generieren")

    # -------------------------------------------------------------------------
    # Subcommand: run (End-to-End)
    # -------------------------------------------------------------------------
    p_run = subparsers.add_parser("run", help="Alle 3 Schritte (Download -> Transkription -> Analyse) in einem Rutsch")
    g_run = p_run.add_mutually_exclusive_group(required=True)
    g_run.add_argument("--search", "-s", type=str, help="Podcast-Name für automatische Suche")
    g_run.add_argument("--feed", "-f", type=str, help="URL des Podcast-RSS-Feeds")
    g_run.add_argument("--file", "-i", type=str, help="Pfad zu lokaler Audiodatei")
    p_run.add_argument("--limit", "-n", type=int, default=1, help="Anzahl der Episoden (Standard: 1)")
    p_run.add_argument("--llm-model", "-m", type=str, default=DEFAULT_OLLAMA_MODEL, help=f"Ollama Modellname (Standard: {DEFAULT_OLLAMA_MODEL})")
    p_run.add_argument("--force", action="store_true", help="Erzwingt Neuberechnung aller Schritte")
    add_common_transcribe_args(p_run)

    # -------------------------------------------------------------------------
    # Top-Level Fallback-Flags für Abwärtskompatibilität:
    # Falls jemand wie früher direkt 'python main.py --search ...' aufruft
    # -------------------------------------------------------------------------
    parser.add_argument("--search", "-s", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--feed", "-f", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--file", "-i", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--limit", "-n", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--llm-model", "-m", type=str, default=DEFAULT_OLLAMA_MODEL, help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    add_common_transcribe_args(parser)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 1. Subcommand: download
    if args.subcommand == "download":
        pipeline = PodcastPipeline()
        pipeline.download_episodes(
            query=args.search,
            feed_url=args.feed,
            limit=args.limit,
        )
        return

    # 2. Subcommand: transcribe
    if args.subcommand == "transcribe":
        pipeline = PodcastPipeline(
            whisper_model=args.whisper_model,
            hf_token=args.hf_token,
            cpu_threads=args.threads,
        )
        audio_path = Path(args.file).resolve() if args.file else None
        pipeline.transcribe_batch(
            audio_file=audio_path,
            language=args.language,
            force=args.force,
        )
        return

    # 3. Subcommand: analyze
    if args.subcommand == "analyze":
        pipeline = PodcastPipeline(
            ollama_model=args.model,
        )
        transcript_path = Path(args.file).resolve() if args.file else None
        pipeline.analyze_batch(
            transcript_file=transcript_path,
            force=args.force,
        )
        return

    # 4. Subcommand: speakers (Optional)
    if args.subcommand == "speakers":
        pipeline = PodcastPipeline(
            ollama_model=args.model,
        )
        transcript_path = Path(args.file).resolve() if args.file else None
        pipeline.analyze_speakers(
            transcript_file=transcript_path,
            force=args.force,
        )
        return

    # 5. Subcommand: run (oder direkte Top-Level Flags)
    is_run = args.subcommand == "run"
    has_legacy_flags = getattr(args, "search", None) or getattr(args, "feed", None) or getattr(args, "file", None)

    if is_run or has_legacy_flags:
        search_arg = getattr(args, "search", None)
        feed_arg = getattr(args, "feed", None)
        file_arg = getattr(args, "file", None)
        limit_arg = getattr(args, "limit", 1)
        force_arg = getattr(args, "force", False)
        lang_arg = getattr(args, "language", None)
        wmodel_arg = getattr(args, "whisper_model", DEFAULT_WHISPER_MODEL)
        omodel_arg = getattr(args, "llm_model", DEFAULT_OLLAMA_MODEL)
        token_arg = getattr(args, "hf_token", HF_TOKEN)
        threads_arg = getattr(args, "threads", 8)

        pipeline = PodcastPipeline(
            whisper_model=wmodel_arg,
            ollama_model=omodel_arg,
            hf_token=token_arg,
            cpu_threads=threads_arg,
        )

        audio_file = Path(file_arg).resolve() if file_arg else None
        pipeline.run_end_to_end(
            query=search_arg,
            feed_url=feed_arg,
            file=audio_file,
            limit=limit_arg,
            language=lang_arg,
            force=force_arg,
        )
        return

    # Kein Befehl angegeben -> Hilfe anzeigen
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
