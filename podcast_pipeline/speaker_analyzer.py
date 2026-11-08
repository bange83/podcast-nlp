import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import ollama

from .config import DEFAULT_OLLAMA_MODEL, REPORTS_DIR

# Häufige deutsche Stopwörter für die Inhaltswort-Filterung
GERMAN_STOPWORDS = frozenset({
    "aber", "als", "am", "an", "auch", "auf", "aus", "bei", "bin", "bis", "bist", "da", "damit",
    "dann", "das", "dass", "dein", "deine", "dem", "den", "der", "des", "dessen", "deshalb",
    "die", "dies", "diese", "dieser", "dieses", "doch", "dort", "du", "durch", "ein", "eine",
    "einem", "einen", "einer", "eines", "er", "es", "etwas", "für", "gab", "ganz", "gehabt",
    "gehen", "gemacht", "gesagt", "gewesen", "hab", "habe", "haben", "hat", "hatte", "hatten",
    "hier", "hin", "hinter", "ich", "ihm", "ihn", "ihr", "ihre", "im", "immer", "in", "ist",
    "ja", "jede", "jedem", "jeden", "jeder", "jedes", "jetzt", "kann", "kannst", "können",
    "könnte", "lassen", "machen", "man", "mehr", "mein", "meine", "mich", "mir", "mit", "muss",
    "musste", "nach", "nein", "nicht", "nichts", "noch", "nun", "nur", "oder", "sehr", "sein",
    "seine", "sich", "sie", "sind", "so", "soll", "sollte", "um", "und", "uns", "unser",
    "unter", "viel", "vom", "von", "vor", "war", "waren", "was", "weil", "weiter", "welche",
    "welcher", "wenn", "wer", "werde", "werden", "wie", "wieder", "wir", "wird", "wo", "wollen",
    "wollte", "wurde", "wurden", "zu", "zum", "zur", "zwar", "zwischen",
})

# Typische Füllwörter / Diskurspartikel im Deutschen
COMMON_FILLER_WORDS = frozenset({
    "quasi", "tatsächlich", "eigentlich", "sozusagen", "irgendwie", "halt", "voll", "krass",
    "alter", "digger", "ey", "weißte", "weißt", "prinzip", "genau", "richtig", "echt",
    "praktisch", "ungelogen", "safe", "brudi", "digga", "nah", "hä",
})


class SpeakerAnalyzer:
    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        output_dir: Path = REPORTS_DIR,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _format_time(seconds: float) -> str:
        secs = int(seconds)
        mins = secs // 60
        rem_secs = secs % 60
        return f"{mins}:{rem_secs:02d} min"

    # =========================================================================
    # 1. Mathematische & Linguistische Statistiken (Deterministisch)
    # =========================================================================

    def compute_statistics(
        self,
        transcript_data: dict[str, Any],
        min_share_percent: float = 5.0,
    ) -> dict[str, Any]:
        """Berechnet Redeanteile, Wortzahlen, Sprechtempo, Monologe und Top-Wörter pro Hauptsprecher."""
        segments = transcript_data.get("segments", [])
        if not segments:
            return {}

        # Sprecher identifizieren (ignoriere UNKNOWN oder reine SPEAKER-Placeholder)
        speaker_segments: dict[str, list[dict[str, Any]]] = {}
        for s in segments:
            spk = s.get("speaker")
            if not spk or spk in {"UNKNOWN", "SPEAKER"}:
                continue
            speaker_segments.setdefault(spk, []).append(s)

        # Gesamtredezeit über alle Segmente ermitteln
        total_audio_speech_time = sum(float(s.get("end", 0)) - float(s.get("start", 0)) for s in segments)

        # Redezeit pro Sprecher berechnen
        speaker_durations = {
            spk: sum(float(s.get("end", 0)) - float(s.get("start", 0)) for s in segs)
            for spk, segs in speaker_segments.items()
        }

        # Filter: Nur Hauptsprecher mit mindestens 5% Redeanteil (filtert Werbung, Jingles und Einspieler heraus)
        main_speakers = [
            spk for spk, dur in sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
            if (dur / total_audio_speech_time * 100 >= min_share_percent)
        ]
        # Falls kein Sprecher 5% erreicht (z.B. bei sehr vielen Gästen), nimm die Top 2
        if len(main_speakers) < 2:
            main_speakers = [
                spk for spk, _ in sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)[:2]
            ]

        stats: dict[str, Any] = {
            "total_speech_time_seconds": total_audio_speech_time,
            "speakers": {},
        }

        # Zusammenhängende Sprecherzüge (Turns / Monologe) ermitteln
        turns: list[dict[str, Any]] = []
        current_turn: dict[str, Any] | None = None

        for s in segments:
            spk = s.get("speaker")
            if not spk or spk not in main_speakers:
                continue
            start_t = float(s.get("start", 0))
            end_t = float(s.get("end", 0))
            text = s.get("text", "").strip()

            if current_turn and current_turn["speaker"] == spk and (start_t - current_turn["end"]) < 2.5:
                # Fortsetzung des Monologs
                current_turn["end"] = end_t
                current_turn["text"] += " " + text
            else:
                if current_turn:
                    turns.append(current_turn)
                current_turn = {
                    "speaker": spk,
                    "start": start_t,
                    "end": end_t,
                    "text": text,
                }
        if current_turn:
            turns.append(current_turn)

        # Pro Sprecher auswerten
        for spk in main_speakers:
            segs = speaker_segments[spk]
            duration = sum(float(s.get("end", 0)) - float(s.get("start", 0)) for s in segs)
            percentage = (duration / total_audio_speech_time * 100) if total_audio_speech_time > 0 else 0.0

            # Alle Wörter extrahieren
            full_spk_text = " ".join(s.get("text", "") for s in segs)
            words = re.findall(r"\b[a-zA-ZäöüÄÖÜß]{2,}\b", full_spk_text)
            total_words = len(words)
            words_lower = [w.lower() for w in words]

            # Sprechgeschwindigkeit (Wörter pro Minute)
            duration_minutes = duration / 60.0
            wpm = round(total_words / duration_minutes, 1) if duration_minutes > 0 else 0.0

            # Monologe dieses Sprechers
            spk_turns = [t for t in turns if t["speaker"] == spk]
            longest_monologue_sec = max([t["end"] - t["start"] for t in spk_turns], default=0.0)
            avg_turn_sec = (sum(t["end"] - t["start"] for t in spk_turns) / len(spk_turns)) if spk_turns else 0.0

            # Füllwörter zählen
            filler_counts = Counter(w for w in words_lower if w in COMMON_FILLER_WORDS)
            top_fillers = [{"word": w, "count": c} for w, c in filler_counts.most_common(6)]

            # Top Inhaltswörter (ohne Stopwörter und ohne Füllwörter)
            content_words = [
                w.title() for w in words_lower
                if w not in GERMAN_STOPWORDS and w not in COMMON_FILLER_WORDS and len(w) > 3
            ]
            content_counts = Counter(content_words)
            top_content_words = [{"word": w, "count": c} for w, c in content_counts.most_common(10)]

            # Lexikalische Vielfalt (Type-Token-Ratio)
            unique_words = len(set(words_lower))
            ttr = round((unique_words / total_words) * 100, 1) if total_words > 0 else 0.0

            stats["speakers"][spk] = {
                "duration_seconds": round(duration, 1),
                "duration_formatted": self._format_time(duration),
                "share_percentage": round(percentage, 1),
                "total_words": total_words,
                "words_per_minute": wpm,
                "turn_count": len(spk_turns),
                "longest_monologue_seconds": round(longest_monologue_sec, 1),
                "longest_monologue_formatted": self._format_time(longest_monologue_sec),
                "avg_turn_seconds": round(avg_turn_sec, 1),
                "lexical_diversity_percent": ttr,
                "top_fillers": top_fillers,
                "top_content_words": top_content_words,
            }

        return stats

    # =========================================================================
    # 2. Psychologische, Rhetorische & Politische Analyse (LLM)
    # =========================================================================

    def profile_speakers(
        self,
        transcript_data: dict[str, Any],
        stats: dict[str, Any],
        title: str | None = None,
    ) -> dict[str, Any]:
        """Lässt das LLM Persönlichkeits-, Sentiment-, Rollen- und Werteprofile für die Sprecher erstellen."""
        speakers = sorted(
            stats.get("speakers", {}).keys(),
            key=lambda s: stats["speakers"][s]["share_percentage"],
            reverse=True,
        )
        if len(speakers) < 2:
            return {}

        segments = transcript_data.get("segments", [])
        episode_title = title or transcript_data.get("audio_file", "Podcast Episode")

        # Für jeden Sprecher die längsten und ausdrucksstärksten Aussagen sammeln
        speaker_samples: dict[str, list[str]] = {spk: [] for spk in speakers}
        for s in segments:
            spk = s.get("speaker")
            if spk in speaker_samples:
                txt = s.get("text", "").strip()
                if len(txt) > 40:
                    speaker_samples[spk].append(txt)

        # Repräsentativen Ausschnitt für das LLM aufbereiten
        sample_sections = []
        for spk in speakers[:2]:
            samples = speaker_samples[spk][:25]
            sample_sections.append(f"### Typische Aussagen von {spk}:\n" + "\n".join(f"- \"{t}\"" for t in samples))

        prompt_samples = "\n\n".join(sample_sections)

        stats_summary = "\n".join(
            f"- {spk}: {data['duration_formatted']} Redezeit ({data['share_percentage']}%), {data['words_per_minute']} WPM, längster Monolog: {data['longest_monologue_formatted']}"
            for spk, data in stats.get("speakers", {}).items()
        )

        prompt = f"""Du bist ein renommierter Medienpsychologe, Rhetorik-Experte und Podcast-Analyst.
Untersuche die beiden Hauptsprecher des Podcasts '{episode_title}'.

Harte Fakten aus der Messung:
{stats_summary}

Hier sind repräsentative Originalzitate der beiden Sprecher aus dieser Episode:
{prompt_samples}

Aufgabe:
Erstelle eine tiefgehende psychologische, rhetorische und inhaltliche Sprecher-Analyse für BEIDE Sprecher ({', '.join(speakers[:2])}).

Analysiere für jeden Sprecher präzise:
1. "archetype": Ein griffiger 3-5 Wörter Titel/Archetyp (z.B. "Der analytisch-spöttische Medienbeobachter", "Der impulsive Straßenphilosoph & Bohemien")
2. "interaction_role": Seine Rolle in dieser Folge (z.B. Themen-Initiator, Moderator, Anekdoten-Erzähler, Schlichter, Provokateur)
3. "emotional_vibe": Dominante Stimmung, Tonalität und Vibe (z.B. melancholisch, zynisch, empathisch, euphorisch, abgekämpft)
4. "recurring_themes": Seine Steckenpferde / Lieblingsthemen, auf die er immer wieder zu sprechen kommt
5. "worldview_and_values": Gesellschaftliche Werte, Haltung & politische Tendenzen (z.B. sozial-liberal, systemkritisch, bürgerlich, kulturpessimistisch)
6. "rhetoric_style": Rhetorischer Stil & Argumentationsmuster (z.B. dozierend, assoziativ, schlagfertig, selbstironisch, anekdotisch)

Beziehungsdynamik:
7. "chemistry_dynamics": Ausführliche Analyse der Beziehungsdynamik zwischen den beiden in dieser Folge (Wer treibt wen an? Wer bremst wen? Wo knistert es humorvoll oder kontrovers?).

Antworte ausschließlich im folgenden JSON-Format ohne Markdown-Codeblöcke:
{{
  "profiles": {{
    "{speakers[0]}": {{
      "archetype": "...",
      "interaction_role": "...",
      "emotional_vibe": "...",
      "recurring_themes": ["...", "..."],
      "worldview_and_values": "...",
      "rhetoric_style": "..."
    }},
    "{speakers[1]}": {{
      "archetype": "...",
      "interaction_role": "...",
      "emotional_vibe": "...",
      "recurring_themes": ["...", "..."],
      "worldview_and_values": "...",
      "rhetoric_style": "..."
    }}
  }},
  "chemistry_dynamics": "Ausführliche Beschreibung der Dynamik zwischen beiden..."
}}
"""
        print(f"[SpeakerAnalyzer] Führe psychologisches Sprecher-Profiling mit '{self.model_name}' durch...")
        try:
            resp = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
            )
            raw = resp["message"]["content"].strip()
            if "{" in raw and "}" in raw:
                raw_json = raw[raw.find("{") : raw.rfind("}") + 1]
                return json.loads(raw_json)
        except Exception as e:
            print(f"[SpeakerAnalyzer] Fehler beim Sprecher-Profiling: {e}")

        return {}

    # =========================================================================
    # 3. Report-Generierung (Markdown & JSON)
    # =========================================================================

    def run_analysis(
        self,
        transcript_data: dict[str, Any],
        title: str | None = None,
    ) -> dict[str, Any]:
        """Führt quantitative Statistik und qualitative LLM-Charakteranalyse durch und speichert Reports."""
        file_stem = Path(transcript_data.get("audio_file", "podcast")).stem
        episode_title = title or file_stem

        # 1. Zahlen & Fakten
        stats = self.compute_statistics(transcript_data)
        if not stats.get("speakers"):
            print("[SpeakerAnalyzer] Keine differenzierten Sprecher im Transkript gefunden.")
            return {}

        # 2. LLM Profiling
        profiles = self.profile_speakers(transcript_data, stats, title=episode_title)

        combined_result = {
            "title": episode_title,
            "statistics": stats,
            "profiles": profiles.get("profiles", {}),
            "chemistry_dynamics": profiles.get("chemistry_dynamics", ""),
        }

        # JSON Report speichern
        json_path = self.output_dir / f"{file_stem}_speakers.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(combined_result, f, ensure_ascii=False, indent=2)

        # Markdown Report speichern
        md_path = self.output_dir / f"{file_stem}_speakers.md"
        self._save_markdown_report(combined_result, md_path)

        print(f"\n✅ Sprecher-Analyse abgeschlossen! Gespeichert unter:\n  -> {json_path}\n  -> {md_path}")
        return combined_result

    def _save_markdown_report(self, data: dict[str, Any], path: Path) -> None:
        """Erzeugt einen ansprechenden Markdown-Bericht über die Sprecher."""
        md = []
        md.append(f"# 🎙️ Sprecher-Analyse: {data.get('title', 'Episode')}\n")

        stats = data.get("statistics", {})
        speakers = sorted(
            stats.get("speakers", {}).keys(),
            key=lambda s: stats["speakers"][s]["share_percentage"],
            reverse=True,
        )
        profiles = data.get("profiles", {})

        if not speakers:
            return

        # 1. Direkte Vergleichstabelle
        md.append("## 📊 Quantitative Sprecher-Statistik\n")
        header = "| Metrik | " + " | ".join(f"**{spk}**" for spk in speakers) + " |"
        sep = "|---| " + " | ".join("---" for _ in speakers) + " |"
        md.append(header)
        md.append(sep)

        md.append("| **Redezeit** | " + " | ".join(f"{stats['speakers'][s]['duration_formatted']} ({stats['speakers'][s]['share_percentage']}%)" for s in speakers) + " |")
        md.append("| **Gesamtwörter** | " + " | ".join(f"{stats['speakers'][s]['total_words']:,} Wörter" for s in speakers) + " |")
        md.append("| **Sprechtempo** | " + " | ".join(f"**{stats['speakers'][s]['words_per_minute']}** WPM" for s in speakers) + " |")
        md.append("| **Längster Monolog** | " + " | ".join(f"{stats['speakers'][s]['longest_monologue_formatted']}" for s in speakers) + " |")
        md.append("| **Durchschn. Redezeit / Turn** | " + " | ".join(f"{stats['speakers'][s]['avg_turn_seconds']}s" for s in speakers) + " |")
        md.append("| **Wortschatz-Vielfalt (TTR)** | " + " | ".join(f"{stats['speakers'][s]['lexical_diversity_percent']}%" for s in speakers) + " |")

        # Top Füllwörter
        def fmt_fillers(s):
            fls = stats['speakers'][s].get('top_fillers', [])
            return ", ".join(f"*{f['word']}* ({f['count']}x)" for f in fls[:4]) if fls else "-"
        md.append("| **Typische Füllwörter** | " + " | ".join(fmt_fillers(s) for s in speakers) + " |")

        # Top Inhaltswörter
        def fmt_content(s):
            cnt = stats['speakers'][s].get('top_content_words', [])
            return ", ".join(f"`{c['word']}`" for c in cnt[:5]) if cnt else "-"
        md.append("| **Häufigste Begriffe** | " + " | ".join(fmt_content(s) for s in speakers) + " |")

        md.append("")

        # 2. Beziehungs- und Gesprächsdynamik
        if data.get("chemistry_dynamics"):
            md.append("## ⚡ Interaktion & Beziehungsdynamik\n")
            md.append(data["chemistry_dynamics"] + "\n")

        # 3. Detaillierte Profile pro Sprecher
        if profiles:
            md.append("## 🎭 Psychologische Sprecher-Profile\n")
            for spk in speakers:
                prof = profiles.get(spk, {})
                if not prof:
                    continue
                md.append(f"### {spk}")
                if "archetype" in prof:
                    md.append(f"> **Archetyp:** *{prof['archetype']}*\n")
                if "interaction_role" in prof:
                    md.append(f"- **Rolle in dieser Folge:** {prof['interaction_role']}")
                if "emotional_vibe" in prof:
                    md.append(f"- **Vibe & Tonalität:** {prof['emotional_vibe']}")
                if "worldview_and_values" in prof:
                    md.append(f"- **Werte & politische Gesinnung:** {prof['worldview_and_values']}")
                if "rhetoric_style" in prof:
                    md.append(f"- **Rhetorischer Stil:** {prof['rhetoric_style']}")
                if prof.get("recurring_themes"):
                    themes_str = ", ".join(f"`{t}`" for t in prof["recurring_themes"])
                    md.append(f"- **Steckenpferde / Themen:** {themes_str}")
                md.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
