import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from faster_whisper import WhisperModel

from .config import (
    TRANSCRIPTS_DIR,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_CPU_THREADS,
    HF_TOKEN
)


class AudioTranscriber:
    def __init__(
        self,
        model_size: str = DEFAULT_WHISPER_MODEL,
        device: str = "cpu",
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        cpu_threads: int = DEFAULT_CPU_THREADS,
        output_dir: Path = TRANSCRIPTS_DIR,
        hf_token: Optional[str] = HF_TOKEN,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token
        self._model: Optional[WhisperModel] = None
        self._diarization_pipeline = None

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

    def _get_diarization_pipeline(self):
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
                if torch.backends.mps.is_available():
                    # Apple Silicon GPU Beschleunigung
                    pipeline.to(torch.device("mps"))
                    print("[Transcriber] Diarization läuft beschleunigt auf Apple Silicon (MPS).")
                self._diarization_pipeline = pipeline
            except Exception as e:
                err_str = str(e)
                print("\n" + "="*70)
                print(f"[Transcriber] ⚠️ FEHLER BEI DER SPRECHERERKENNUNG (PyAnnote):")
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

    def _run_diarization(self, audio_path: Path):
        """Führt Sprechererkennung auf der Audiodatei durch."""
        pipeline = self._get_diarization_pipeline()
        if not pipeline:
            return None

        print(f"[Transcriber] Führe Sprechertrennung (Diarization) für {audio_path.name} durch...")
        try:
            diarization = pipeline(str(audio_path))
            return diarization
        except Exception as e:
            print(f"[Transcriber] Fehler bei Diarization: {e}")
            return None

    @staticmethod
    def _assign_speaker_to_segment(seg_start: float, seg_end: float, diarization) -> str:
        """Findet den dominanten Sprecher für einen gegebenen Zeitbereich."""
        if not diarization:
            return "UNKNOWN"

        try:
            from pyannote.core import Segment
            segment = Segment(seg_start, seg_end)
            cropped = diarization.crop(segment)

            speaker_durations = {}
            for turn, _, speaker in cropped.itertracks(yield_label=True):
                duration = turn.length
                speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration

            if speaker_durations:
                return max(speaker_durations, key=speaker_durations.get)

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

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """Transkribiert eine Audiodatei und ordnet optional Sprecher (Jan vs Olli) zu."""
        stem = audio_path.stem
        json_path = self.output_dir / f"{stem}.json"
        txt_path = self.output_dir / f"{stem}.txt"

        if not force_recompute and json_path.exists() and txt_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)

            # Wenn das Transkript bereits da ist, aber noch keine Sprecher hat:
            if not existing_data.get("has_diarization") and self.hf_token:
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

                    print(f"[Transcriber] Sprecher erfolgreich zugeordnet!")
                    return existing_data

            print(f"[Transcriber] Transkript existiert bereits: {json_path.name}")
            return existing_data

        # 1. Optionale Diarization (Sprechererkennung)
        diarization = self._run_diarization(audio_path)

        # 2. Whisper Transkription
        print(f"[Transcriber] Starte Transkription von: {audio_path.name}")
        segments_gen, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True
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
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]"
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
                    "text": text
                }
                segments.append(seg_dict)

                # Formatierte Dialogzeile
                if speaker != "UNKNOWN":
                    dialogue_lines.append(f"[{start}s - {end}s] {speaker}: {text}")
                else:
                    dialogue_lines.append(f"[{start}s - {end}s] {text}")

            if pbar.n < total_seconds:
                pbar.update(total_seconds - pbar.n)

        full_text = "\n".join(dialogue_lines)

        result_data = {
            "audio_file": audio_path.name,
            "language": detected_lang,
            "language_probability": lang_prob,
            "duration_seconds": duration,
            "has_diarization": diarization is not None,
            "full_text": full_text,
            "segments": segments
        }

        # Speichern als JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        # Speichern als Textdatei
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"[Transcriber] Fertig! Gespeichert unter:\n  -> {json_path}\n  -> {txt_path}")
        return result_data
