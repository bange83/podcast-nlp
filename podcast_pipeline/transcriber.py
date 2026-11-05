import json
from pathlib import Path
from typing import Dict, Any, Optional
from faster_whisper import WhisperModel

from .config import TRANSCRIPTS_DIR, DEFAULT_WHISPER_MODEL, DEFAULT_COMPUTE_TYPE


class AudioTranscriber:
    def __init__(
        self,
        model_size: str = DEFAULT_WHISPER_MODEL,
        device: str = "cpu",
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        output_dir: Path = TRANSCRIPTS_DIR,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model: Optional[WhisperModel] = None

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            print(f"[Transcriber] Lade lokales Whisper-Modell '{self.model_size}' auf {self.device} ({self.compute_type})...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """Transkribiert eine Audiodatei und speichert Text + Zeitstempel als JSON und TXT ab."""
        stem = audio_path.stem
        json_path = self.output_dir / f"{stem}.json"
        txt_path = self.output_dir / f"{stem}.txt"

        if not force_recompute and json_path.exists() and txt_path.exists():
            print(f"[Transcriber] Transkript existiert bereits: {json_path.name}")
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)

        print(f"[Transcriber] Starte Transkription von: {audio_path.name}")
        segments_gen, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True  # Filtert Stille heraus für schnellere Verarbeitung
        )

        detected_lang = info.language
        lang_prob = round(info.language_probability, 3)
        duration = round(info.duration, 2)
        print(f"[Transcriber] Erkannte Sprache: '{detected_lang}' (Wahrscheinlichkeit: {lang_prob}), Dauer: {duration}s")

        segments = []
        full_text_parts = []

        for segment in segments_gen:
            seg_dict = {
                "id": segment.id,
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            }
            segments.append(seg_dict)
            full_text_parts.append(segment.text.strip())

        full_text = " ".join(full_text_parts)

        result_data = {
            "audio_file": audio_path.name,
            "language": detected_lang,
            "language_probability": lang_prob,
            "duration_seconds": duration,
            "full_text": full_text,
            "segments": segments
        }

        # Speichern als JSON (strukturiert mit Timestamps)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        # Speichern als reine Textdatei (Lesbarkeit)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"[Transcriber] Fertig! Gespeichert unter:\n  -> {json_path}\n  -> {txt_path}")
        return result_data
