import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel
from tqdm import tqdm

from .config import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_CPU_THREADS,
    DEFAULT_TRANSCRIBER_BACKEND,
    DEFAULT_WHISPER_MODEL,
    HF_TOKEN,
    MLX_WHISPER_MODELS,
    TRANSCRIPTS_DIR,
)


def is_mlx_supported() -> bool:
    """Prüft, ob Apple Silicon (macOS arm64) und mlx-whisper verfügbar sind."""
    if sys.platform != "darwin" or platform.machine().lower() != "arm64":
        return False
    try:
        import mlx_whisper  # noqa: F401
        return True
    except ImportError:
        return False


class AudioTranscriber:
    def __init__(
        self,
        model_size: str = DEFAULT_WHISPER_MODEL,
        backend: str = DEFAULT_TRANSCRIBER_BACKEND,
        device: str = "cpu",
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        cpu_threads: int = DEFAULT_CPU_THREADS,
        output_dir: Path = TRANSCRIPTS_DIR,
        hf_token: str | None = HF_TOKEN,
    ):
        self.model_size = model_size
        self.requested_backend = backend.lower().strip()
        self.backend = self._resolve_backend(self.requested_backend)
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token or HF_TOKEN
        self._model: WhisperModel | None = None
        self._diarization_pipeline: Any = None

    def _resolve_backend(self, backend: str) -> str:
        """Ermittelt das optimale Transkriptions-Backend basierend auf Hardware und Nutzeranforderung."""
        if backend == "auto":
            if is_mlx_supported():
                print("[Transcriber] 🚀 Apple Silicon erkannt: Nutze hardwarebeschleunigtes 'mlx-whisper' (Metal GPU).")
                return "mlx"
            else:
                print("[Transcriber] Standard-Backend 'faster-whisper' (CPU/CUDA) wird genutzt.")
                return "faster-whisper"
        elif backend == "mlx":
            if not is_mlx_supported():
                if sys.platform != "darwin" or platform.machine().lower() != "arm64":
                    raise RuntimeError("Backend 'mlx' ist nur auf macOS mit Apple Silicon (arm64) verfügbar.")
                raise RuntimeError("Backend 'mlx' ausgewählt, aber 'mlx-whisper' ist nicht installiert. Bitte mit 'uv sync' installieren.")
            print("[Transcriber] 🚀 Explizites Backend 'mlx' aktiviert (Metal GPU).")
            return "mlx"
        elif backend in ("faster-whisper", "faster_whisper"):
            print("[Transcriber] Explizites Backend 'faster-whisper' aktiviert.")
            return "faster-whisper"
        else:
            print(f"[Transcriber] Unbekanntes Backend '{backend}', nutze Standard 'faster-whisper'.")
            return "faster-whisper"

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            print(f"[Transcriber] Lade lokales Whisper-Modell '{self.model_size}' ({self.compute_type}, {self.cpu_threads} CPU-Threads)...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=2
            )
        return self._model

    def _get_diarization_pipeline(self) -> Any:
        """Initialisiert die PyAnnote Diarization Pipeline via HuggingFace."""
        if not self.hf_token:
            return None
        if self._diarization_pipeline is None:
            try:
                import torch
                from pyannote.audio import Pipeline
                print("[Transcriber] Lade PyAnnote Speaker Diarization Pipeline von HuggingFace...")
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=self.hf_token
                )
                if pipeline is not None and torch.backends.mps.is_available():
                    # Apple Silicon GPU Beschleunigung
                    pipeline.to(torch.device("mps"))
                    print("[Transcriber] Diarization läuft beschleunigt auf Apple Silicon (MPS).")
                self._diarization_pipeline = pipeline
            except Exception as e:
                err_str = str(e)
                print("\n" + "="*70)
                print("[Transcriber] ⚠️ FEHLER BEI DER SPRECHERERKENNUNG (PyAnnote):")
                if "403" in err_str or "gated" in err_str:
                    print("Der Hugging Face Token hat noch keinen Zugriff auf alle PyAnnote-Modelle.")
                    print("Bitte stellen Sie sicher, dass Sie bei Hugging Face eingeloggt sind")
                    print("und die Nutzungsbedingungen für diese 3 Modelle akzeptiert haben:")
                    print("  1. 👉 https://huggingface.co/pyannote/speaker-diarization-community-1")
                    print("  2. 👉 https://huggingface.co/pyannote/speaker-diarization-3.1")
                    print("  3. 👉 https://huggingface.co/pyannote/segmentation-3.0")
                else:
                    print(f"Fehler: {err_str}")
                print("="*70 + "\n")
                self._diarization_pipeline = False
        return self._diarization_pipeline if self._diarization_pipeline is not False else None

    def _run_diarization(self, audio_path: Path) -> Any:
        """Führt Sprechererkennung auf der Audiodatei durch und cacht das RTTM-Ergebnis."""
        if audio_path.parent.name == "audio":
            transcripts_dir = audio_path.parent.parent / "transcripts"
        else:
            transcripts_dir = self.output_dir
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        rttm_path = transcripts_dir / f"{audio_path.stem}.rttm"
        if rttm_path.exists():
            try:
                from pyannote.core import Annotation, Segment
                annotation = Annotation(uri=audio_path.name)
                with open(rttm_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 8 and parts[0] == "SPEAKER":
                            start = float(parts[3])
                            dur = float(parts[4])
                            spk = parts[7]
                            annotation[Segment(start, start + dur)] = spk
                print(f"[Transcriber] Lade gecachte Sprechererkennung: {rttm_path.name}")
                return annotation
            except Exception as e:
                print(f"[Transcriber] Konnte RTTM nicht laden ({e}), rechne neu...")

        pipeline = self._get_diarization_pipeline()
        if not pipeline:
            return None

        print(f"[Transcriber] Führe Sprechertrennung (Diarization) für {audio_path.name} durch...")
        temp_wav: Path | None = None
        try:
            # PyAnnote benötigt für exaktes Chunking 16kHz Mono PCM Audio.
            # Bei MP3s (insb. Stereo oder VBR) wirft der direkte MP3-Decoder oft Sample-Pufferfehler.
            if audio_path.suffix.lower() != ".wav":
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    temp_wav = Path(tf.name)
                # Schnelle Konvertierung via ffmpeg (~2s für 70 min Audio)
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(audio_path),
                    "-ac", "1", "-ar", "16000",
                    str(temp_wav),
                ]
                subprocess.run(cmd, check=True)
                diarize_input = str(temp_wav)
            else:
                diarize_input = str(audio_path)

            res: Any = pipeline(diarize_input)
            # PyAnnote 3.1+ gibt DiarizeOutput zurück
            annotation: Any = getattr(res, "speaker_diarization", res)

            # Als RTTM für sofortige Wiederverwendung speichern
            try:
                with open(rttm_path, "w", encoding="utf-8") as f:
                    annotation.write_rttm(f)
                print(f"[Transcriber] Diarization-Cache gespeichert: {rttm_path.name}")
            except Exception:
                pass

            return annotation
        except Exception as e:
            print(f"[Transcriber] Fehler bei Diarization: {e}")
            return None
        finally:
            if temp_wav and temp_wav.exists():
                try:
                    temp_wav.unlink()
                except Exception:
                    pass

    @staticmethod
    def _assign_speaker_to_segment(seg_start: float, seg_end: float, diarization: Any) -> str:
        """Findet den dominanten Sprecher für einen gegebenen Zeitbereich."""
        if not diarization:
            return "UNKNOWN"

        try:
            # Falls noch DiarizeOutput übergeben wurde:
            diarization = getattr(diarization, "speaker_diarization", diarization)
            from pyannote.core import Segment
            segment = Segment(seg_start, seg_end)
            cropped = diarization.crop(segment)

            speaker_durations: dict[str, float] = {}
            for turn, _, speaker in cropped.itertracks(yield_label=True):
                dur = turn.end - turn.start
                speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + dur

            if speaker_durations:
                return max(speaker_durations, key=lambda spk: speaker_durations[spk])

            # Fallback 1: Prüfe den Mittelpunkt des Segments
            mid_point = (seg_start + seg_end) / 2
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= mid_point <= turn.end:
                    return speaker

            # Fallback 2: Nächster zeitlicher Sprecher innerhalb von 1 Sekunde
            closest_speaker = None
            min_dist = float("inf")
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                dist = min(abs(turn.start - seg_end), abs(turn.end - seg_start))
                if dist < min_dist and dist < 1.5:
                    min_dist = dist
                    closest_speaker = speaker

            if closest_speaker:
                return closest_speaker

        except Exception:
            pass

        return "UNKNOWN"

    def _transcribe_mlx(
        self,
        audio_path: Path,
        language: str | None = None,
        diarization: Any = None,
    ) -> dict[str, Any]:
        """Transkribiert mit Apple Silicon Metal GPU Beschleunigung via mlx-whisper."""
        import mlx_whisper

        repo_or_path = MLX_WHISPER_MODELS.get(self.model_size, self.model_size)
        print(f"[Transcriber] ⚡ MLX Metal GPU Beschleunigung aktiv (Modell: '{repo_or_path}')...")

        decode_kwargs: dict[str, Any] = {}
        if language:
            decode_kwargs["language"] = language

        t0 = time.time()
        res = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=repo_or_path,
            **decode_kwargs,
        )
        elapsed = time.time() - t0

        res_dict: dict[str, Any] = res if isinstance(res, dict) else {}
        raw_segments_val = res_dict.get("segments", [])
        raw_segments: list[Any] = raw_segments_val if isinstance(raw_segments_val, list) else []
        detected_lang = res_dict.get("language", language or "unknown")

        segments: list[dict[str, Any]] = []
        dialogue_lines: list[str] = []

        for idx, seg in enumerate(raw_segments):
            seg_dict_raw: dict[str, Any] = seg if isinstance(seg, dict) else {}
            start = round(float(seg_dict_raw.get("start", 0.0)), 2)
            end = round(float(seg_dict_raw.get("end", 0.0)), 2)
            text = str(seg_dict_raw.get("text", "")).strip()
            speaker = self._assign_speaker_to_segment(start, end, diarization)

            seg_dict = {
                "id": seg_dict_raw.get("id", idx),
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
            }
            segments.append(seg_dict)

            if speaker != "UNKNOWN":
                dialogue_lines.append(f"[{start}s - {end}s] {speaker}: {text}")
            else:
                dialogue_lines.append(f"[{start}s - {end}s] {text}")

        full_text = "\n".join(dialogue_lines)
        duration = round(segments[-1]["end"], 2) if segments else 0.0
        speedup = (duration / elapsed) if elapsed > 0 and duration > 0 else 0.0
        mins = int(duration // 60)
        secs = int(duration % 60)
        print(f"[Transcriber] ⚡ MLX Metal Transkription abgeschlossen in {elapsed:.1f}s für {mins}m {secs}s Audio ({speedup:.1f}x Echtzeit)!")

        return {
            "audio_file": audio_path.name,
            "language": detected_lang,
            "language_probability": 1.0,
            "duration_seconds": duration,
            "has_diarization": diarization is not None,
            "backend": "mlx",
            "full_text": full_text,
            "segments": segments,
        }

    def _transcribe_faster_whisper(
        self,
        audio_path: Path,
        language: str | None = None,
        diarization: Any = None,
    ) -> dict[str, Any]:
        """Transkribiert mit CTranslate2 (faster-whisper) auf CPU oder CUDA."""
        print(f"[Transcriber] Starte faster-whisper Transkription von: {audio_path.name}")
        segments_gen, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        detected_lang = info.language
        lang_prob = round(info.language_probability, 3)
        duration = round(info.duration, 2)
        total_seconds = int(duration)
        print(f"[Transcriber] Erkannte Sprache: '{detected_lang}' ({lang_prob*100:.1f}%), Gesamtdauer: {total_seconds // 60}m {total_seconds % 60}s")

        segments = []
        dialogue_lines = []
        last_reported_end = 0.0

        with tqdm(
            total=total_seconds,
            unit="s",
            desc="[Transcriber] Fortschritt",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]",
        ) as pbar:
            for segment in segments_gen:
                start = round(segment.start, 2)
                end = round(segment.end, 2)
                text = segment.text.strip()

                advance = int(end) - int(last_reported_end)
                if advance > 0:
                    pbar.update(advance)
                    last_reported_end = end

                speaker = self._assign_speaker_to_segment(start, end, diarization)

                seg_dict = {
                    "id": segment.id,
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": text,
                }
                segments.append(seg_dict)

                if speaker != "UNKNOWN":
                    dialogue_lines.append(f"[{start}s - {end}s] {speaker}: {text}")
                else:
                    dialogue_lines.append(f"[{start}s - {end}s] {text}")

            if pbar.n < total_seconds:
                pbar.update(total_seconds - pbar.n)

        full_text = "\n".join(dialogue_lines)

        return {
            "audio_file": audio_path.name,
            "language": detected_lang,
            "language_probability": lang_prob,
            "duration_seconds": duration,
            "has_diarization": diarization is not None,
            "backend": "faster-whisper",
            "full_text": full_text,
            "segments": segments,
        }

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        force_recompute: bool = False,
    ) -> dict[str, Any]:
        """Transkribiert eine Audiodatei und ordnet optional Sprecher (z.B. Jan vs Olli) zu."""
        if audio_path.parent.name == "audio":
            out_dir = audio_path.parent.parent / "transcripts"
        else:
            out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = audio_path.stem
        json_path = out_dir / f"{stem}.json"
        txt_path = out_dir / f"{stem}.txt"

        if not force_recompute and json_path.exists() and txt_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)

            # Wenn das Transkript bereits da ist, aber noch keine echten Sprecher hat:
            speakers = {s.get("speaker") for s in existing_data.get("segments", []) if s.get("speaker")}
            needs_diarization = (not existing_data.get("has_diarization")) or (speakers == {"UNKNOWN"}) or ("UNKNOWN" in speakers and len(speakers) == 1)

            if needs_diarization and self.hf_token:
                print(f"[Transcriber] ✨ Nutze bestehendes Whisper-Transkript ({json_path.name})!")
                print("[Transcriber] Starte nur die fehlende Sprechererkennung (Diarization)...")
                diarization = self._run_diarization(audio_path)
                if diarization:
                    segments = existing_data.get("segments", [])
                    dialogue_lines = []
                    for seg in segments:
                        spk = self._assign_speaker_to_segment(seg["start"], seg["end"], diarization)
                        seg["speaker"] = spk
                        if spk != "UNKNOWN":
                            dialogue_lines.append(f"[{seg['start']}s - {seg['end']}s] {spk}: {seg['text']}")
                        else:
                            dialogue_lines.append(f"[{seg['start']}s - {seg['end']}s] {seg['text']}")

                    existing_data["has_diarization"] = True
                    existing_data["full_text"] = "\n".join(dialogue_lines)
                    existing_data["segments"] = segments

                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=2)
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(existing_data["full_text"])

                    print("[Transcriber] Sprecher erfolgreich zugeordnet!")
                    return existing_data

            print(f"[Transcriber] Transkript existiert bereits: {json_path.name}")
            return existing_data

        # 1. Optionale Diarization (Sprechererkennung)
        diarization = self._run_diarization(audio_path)

        # 2. Whisper Transkription via gewähltem Backend (mit automatischem Fallback)
        print(f"[Transcriber] Starte Transkription von: {audio_path.name} (Backend: {self.backend})")
        if self.backend == "mlx":
            try:
                result_data = self._transcribe_mlx(audio_path, language=language, diarization=diarization)
            except Exception as e:
                print(f"[Transcriber] ⚠️ Fehler bei MLX-Transkription: {e}")
                print("[Transcriber] Wechsle automatisch auf Fallback 'faster-whisper'...")
                result_data = self._transcribe_faster_whisper(audio_path, language=language, diarization=diarization)
        else:
            result_data = self._transcribe_faster_whisper(audio_path, language=language, diarization=diarization)

        # Speichern als JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        # Speichern als Textdatei
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result_data["full_text"])

        print(f"[Transcriber] Fertig! Gespeichert unter:\n  -> {json_path}\n  -> {txt_path}")
        return result_data
