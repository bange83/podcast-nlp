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

# Alltägliche Konversationsverben und -partikel für gesprochenes Deutsch
SPOKEN_CONVERSATIONAL_WORDS = frozenset({
    "also", "sagen", "gesagt", "sagte", "sagst", "glaube", "glauben", "glaubst",
    "weiß", "wissen", "weißt", "schon", "mal", "eben", "einfach", "wirklich",
    "immer", "sehen", "gesehen", "siehst", "geht", "ging", "gegangen", "denken",
    "gedacht", "denkst", "meinen", "gemeint", "meinst", "irgendwann", "überhaupt",
    "ganze", "ganzen", "ganzer", "ganzes", "natürlich", "dinge", "ding", "finden",
    "gefunden", "findest", "kommen", "kommt", "kam", "gekommen", "menschen", "leute",
    "wäre", "geben", "gibt", "gab", "gegeben", "lassen", "lässt", "machen", "macht",
    "gemacht", "erst", "wieder", "sondern", "kein", "keine", "keinen", "keinem",
    "keiner", "hast", "hatte", "hatten", "heißt", "über", "alles", "neue", "neuen",
    "neuer", "neues", "viele", "vielen", "vieles", "große", "großen", "großer",
    "großes", "zwei", "drei", "vier", "fünf", "zehn", "hundert", "tausend", "paar",
    "gibt's", "gabs", "fragen", "gefragt", "jahr", "jahre", "jahren", "tag", "tage",
    "tagen", "zeit", "fall", "stelle", "grund", "beispiel", "prozent", "gucken",
    "guckt", "guck", "schauen", "steht", "stehen", "nehmen", "nimmt", "wollten",
})

QUESTION_REGEX = re.compile(
    r'(\?|\b(warum|wieso|weshalb|inwiefern|woher|wohin)\b|\b(glaubst du|denkst du|hast du|meinst du|ist das nicht|kann man|findest du|weißt du)\b)',
    re.I
)


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

    @staticmethod
    def _compute_mtld(words: list[str], threshold: float = 0.72) -> float:
        """Berechnet das längeninvariante Measure of Textual Lexical Diversity (MTLD)."""
        if len(words) < 10:
            return 0.0

        def _calc_factors(word_list: list[str]) -> float:
            factors = 0.0
            types = set()
            token_count = 0
            for w in word_list:
                types.add(w)
                token_count += 1
                ttr = len(types) / token_count
                if ttr <= threshold:
                    factors += 1.0
                    types = set()
                    token_count = 0
            if token_count > 0:
                excess_ttr = (1.0 - (len(types) / token_count)) / (1.0 - threshold)
                factors += excess_ttr
            return len(word_list) / factors if factors > 0 else float(len(word_list))

        fwd = _calc_factors(words)
        bwd = _calc_factors(words[::-1])
        return round((fwd + bwd) / 2.0, 1)

    @staticmethod
    def _compute_interruptions(
        segments: list[dict[str, Any]],
        rttm_path: Path | None,
        main_speakers: list[str],
        speaker_mapping: dict[str, str],
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Ermittelt echte Unterbrechungen (Overlaps) aus RTTM-Audiodaten oder Segmenten."""
        made: dict[str, int] = {spk: 0 for spk in main_speakers}
        suffered: dict[str, int] = {spk: 0 for spk in main_speakers}

        if rttm_path and rttm_path.exists():
            tracks: list[tuple[float, float, str]] = []
            try:
                with open(rttm_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 8 and parts[0] == "SPEAKER":
                            s = float(parts[3])
                            dur = float(parts[4])
                            raw_spk = parts[7]
                            resolved = speaker_mapping.get(raw_spk, raw_spk)
                            if resolved in main_speakers:
                                tracks.append((s, s + dur, resolved))

                for i in range(len(tracks)):
                    s1, e1, spk1 = tracks[i]
                    for j in range(i + 1, min(i + 8, len(tracks))):
                        s2, e2, spk2 = tracks[j]
                        if spk1 != spk2 and s2 < (e1 - 0.25) and s2 > (s1 + 0.3):
                            made[spk2] = made.get(spk2, 0) + 1
                            suffered[spk1] = suffered.get(spk1, 0) + 1
                return made, suffered
            except Exception:
                pass

        # Fallback über Transkriptsegmente
        for i in range(1, len(segments)):
            prev_s = segments[i - 1]
            curr_s = segments[i]
            spk_prev = prev_s.get("speaker")
            spk_curr = curr_s.get("speaker")
            if (
                spk_prev in main_speakers
                and spk_curr in main_speakers
                and spk_prev != spk_curr
            ):
                dt = float(curr_s.get("start", 0)) - float(prev_s.get("end", 0))
                if dt < -0.2:
                    made[spk_curr] = made.get(spk_curr, 0) + 1
                    suffered[spk_prev] = suffered.get(spk_prev, 0) + 1

        return made, suffered

    # =========================================================================
    # 1. Mathematische & Linguistische Statistiken (Deterministisch)
    # =========================================================================

    def compute_statistics(
        self,
        transcript_data: dict[str, Any],
        min_share_percent: float = 5.0,
        rttm_path: Path | None = None,
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

        # Filter: Nur Hauptsprecher mit mindestens 5% Redeanteil
        main_speakers = [
            spk for spk, dur in sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
            if (dur / total_audio_speech_time * 100 >= min_share_percent)
        ]
        if len(main_speakers) < 2:
            main_speakers = [
                spk for spk, _ in sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)[:2]
            ]

        # Unterbrechungen ermitteln
        spk_mapping = transcript_data.get("speaker_mapping", {})
        interruptions_made, interruptions_suffered = self._compute_interruptions(
            segments, rttm_path, main_speakers, spk_mapping
        )

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

            # Fragen erkennen
            spk_texts = [s.get("text", "") for s in segs]
            question_count = sum(1 for t in spk_texts if QUESTION_REGEX.search(t))
            question_pct = round((question_count / len(spk_texts) * 100), 1) if spk_texts else 0.0

            # Füllwörter zählen
            filler_counts = Counter(w for w in words_lower if w in COMMON_FILLER_WORDS)
            top_fillers = [{"word": w, "count": c} for w, c in filler_counts.most_common(6)]

            # Top Inhaltswörter / Substantive (ohne Stopwörter, ohne Füllwörter und ohne Alltagsverben)
            filter_set = GERMAN_STOPWORDS | COMMON_FILLER_WORDS | SPOKEN_CONVERSATIONAL_WORDS
            content_words = [
                w.capitalize() for w in words
                if w.lower() not in filter_set and len(w) > 3
            ]
            content_counts = Counter(content_words)
            top_content_words = [{"word": w, "count": c} for w, c in content_counts.most_common(8)]

            # Lexikalische Vielfalt: Längenunabhängiges MTLD (Gesamt & Inhaltswörter) & klassische TTR
            mtld_total = self._compute_mtld(words_lower)
            content_words_for_mtld = [w for w in words_lower if w not in GERMAN_STOPWORDS]
            mtld_content = self._compute_mtld(content_words_for_mtld)
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
                "question_count": question_count,
                "question_percentage": question_pct,
                "interruptions_made": interruptions_made.get(spk, 0),
                "interruptions_suffered": interruptions_suffered.get(spk, 0),
                "mtld_score": mtld_total,
                "mtld_content_score": mtld_content,
                "lexical_diversity_percent": ttr,
                "top_fillers": top_fillers,
                "top_content_words": top_content_words,
            }

        return stats

    # =========================================================================
    # 2. Interaktions-Klassifikation & Diskurs-Alignment (LLM Micro-Batching)
    # =========================================================================

    def _classify_interactions(
        self,
        segments: list[dict[str, Any]],
        main_speakers: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Klassifiziert Sprecherwechsel & Unterbrechungen via Micro-Batching mit dem lokalen LLM."""
        takeovers: list[dict[str, Any]] = []
        for i in range(1, len(segments)):
            prev_s = segments[i - 1]
            curr_s = segments[i]
            spk_prev = prev_s.get("speaker")
            spk_curr = curr_s.get("speaker")
            if (
                spk_prev in main_speakers
                and spk_curr in main_speakers
                and spk_prev != spk_curr
            ):
                dt = float(curr_s.get("start", 0)) - float(prev_s.get("end", 0))
                # Schneller Wechsel / Unterbrechung / Latching
                if dt < 0.6:
                    prev_txt = prev_s.get("text", "").strip()
                    curr_txt = curr_s.get("text", "").strip()
                    if prev_txt and curr_txt:
                        takeovers.append({
                            "id": len(takeovers) + 1,
                            "vorredner": spk_prev,
                            "vorredner_text": prev_txt[-120:],
                            "einwurf_sprecher": spk_curr,
                            "einwurf_text": curr_txt[:120],
                            "dt": round(dt, 2),
                        })

        if not takeovers:
            return {
                spk: {
                    "supportive_pct": 50.0,
                    "steering_pct": 30.0,
                    "disruptive_pct": 20.0,
                    "listening_score": 4.0,
                    "evaluated_count": 0,
                }
                for spk in main_speakers
            }

        # Bis zu 30 repräsentative Takeovers gleichmäßig samplen für blitzschnelle Inferenz
        max_samples = 30
        if len(takeovers) > max_samples:
            step = len(takeovers) / max_samples
            sampled_takeovers = [takeovers[int(i * step)] for i in range(max_samples)]
            for idx, item in enumerate(sampled_takeovers, 1):
                item["id"] = idx
        else:
            sampled_takeovers = takeovers

        batch_size = 15
        evaluations: list[dict[str, Any]] = []
        print(f"[SpeakerAnalyzer] Klassifiziere {len(sampled_takeovers)} Interaktionen via LLM-Micro-Batching...")

        for b_idx in range(0, len(sampled_takeovers), batch_size):
            batch = sampled_takeovers[b_idx : b_idx + batch_size]
            batch_prompt = f"""Du bist ein präziser Konversations- und Diskursanalyst.
Analysiere die folgenden {len(batch)} Sprecherwechsel und Unterbrechungen:
{json.dumps(batch, ensure_ascii=False, indent=2)}

Bewerte jeden Wechsel für den 'einwurf_sprecher':
1. "style":
   - "supportive" (Zustimmung, Bestärkung, Lachen, Gag oder Satz vervollständigen)
   - "steering" (Nachhaken, investigative Frage stellen, Thema lenken, neue Facette)
   - "disruptive" (Gegenargument, Abwürgen/Unterbrechung, Redezeit an sich reißen)
2. "alignment": Zahl von 1 bis 5 (1 = ignoriert Vorredner komplett / harter Themenwechsel, 3 = lockerer Bezug, 5 = geht direkt und tief auf das Argument des Vorredners ein)

Antworte ausschließlich im folgenden JSON-Format ohne Markdown-Backticks:
{{
  "evaluations": [
    {{"id": 1, "style": "supportive", "alignment": 4}}
  ]
}}
"""
            try:
                resp = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": batch_prompt}],
                    options={"temperature": 0.1},
                )
                raw = resp["message"]["content"].strip()
                if "{" in raw and "}" in raw:
                    raw_json = raw[raw.find("{") : raw.rfind("}") + 1]
                    parsed = json.loads(raw_json, strict=False)
                    evaluations.extend(parsed.get("evaluations", []))
            except Exception as e:
                print(f"[SpeakerAnalyzer] Hinweis: Micro-Batch {b_idx // batch_size + 1} nutzte Heuristik-Fallback ({e})")

        eval_by_id = {e.get("id"): e for e in evaluations if "id" in e}
        speaker_stats: dict[str, dict[str, Any]] = {
            spk: {"supportive": 0, "steering": 0, "disruptive": 0, "alignments": []}
            for spk in main_speakers
        }

        for item in sampled_takeovers:
            spk = item["einwurf_sprecher"]
            if spk not in speaker_stats:
                continue

            ev = eval_by_id.get(item["id"])
            if ev:
                st = str(ev.get("style", "supportive")).lower()
                if "support" in st or "agree" in st:
                    style = "supportive"
                elif "steer" in st or "clarif" in st or "quest" in st:
                    style = "steering"
                else:
                    style = "disruptive"
                align = float(ev.get("alignment", 4.0))
            else:
                txt = item["einwurf_text"].lower()
                if any(w in txt[:25] for w in ("genau", "richtig", "ja", "voll", "klar", "stimmt", "so ist")):
                    style = "supportive"
                    align = 4.5
                elif "?" in txt or any(w in txt[:25] for w in ("warum", "aber", "wie", "was", "wieso")):
                    style = "steering"
                    align = 4.0
                else:
                    style = "disruptive"
                    align = 3.2

            speaker_stats[spk][style] += 1
            speaker_stats[spk]["alignments"].append(align)

        result: dict[str, dict[str, Any]] = {}
        for spk, data in speaker_stats.items():
            total_evals = data["supportive"] + data["steering"] + data["disruptive"]
            if total_evals > 0:
                s_pct = round((data["supportive"] / total_evals) * 100, 1)
                st_pct = round((data["steering"] / total_evals) * 100, 1)
                d_pct = round((data["disruptive"] / total_evals) * 100, 1)
                avg_align = round(sum(data["alignments"]) / len(data["alignments"]), 1)
            else:
                s_pct, st_pct, d_pct, avg_align = 40.0, 40.0, 20.0, 4.0

            result[spk] = {
                "supportive_pct": s_pct,
                "steering_pct": st_pct,
                "disruptive_pct": d_pct,
                "listening_score": avg_align,
                "evaluated_count": total_evals,
            }

        return result

    # =========================================================================
    # 3. Psychologische, Rhetorische & Politische Analyse (LLM)
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
            f"- {spk}: {data['duration_formatted']} Redezeit ({data['share_percentage']}%), "
            f"{data['words_per_minute']} WPM, {data.get('question_count', 0)} Fragen ({data.get('question_percentage', 0.0)}%), "
            f"Unterbrechungen: {data.get('interruptions_made', 0)}x erteilt / {data.get('interruptions_suffered', 0)}x erlitten, "
            f"Wortschatz-MTLD: {data.get('mtld_score', 0)}, "
            f"Häufigste thematische Begriffe: {', '.join(w['word'] for w in data.get('top_content_words', [])[:6])}"
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
                return json.loads(raw_json, strict=False)
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
        output_dir: Path | None = None,
        rttm_path: Path | None = None,
    ) -> dict[str, Any]:
        """Führt quantitative Statistik und qualitative LLM-Charakteranalyse durch und speichert Reports."""
        file_stem = Path(transcript_data.get("audio_file", "podcast")).stem
        episode_title = title or file_stem

        # Falls rttm_path nicht übergeben wurde, in transcripts-Verzeichnis nachsehen
        resolved_rttm = rttm_path
        if not resolved_rttm and output_dir:
            candidate = output_dir.parent / "transcripts" / f"{file_stem}.rttm"
            if candidate.exists():
                resolved_rttm = candidate

        # 1. Zahlen & Fakten
        stats = self.compute_statistics(transcript_data, rttm_path=resolved_rttm)
        if not stats.get("speakers"):
            print("[SpeakerAnalyzer] Keine differenzierten Sprecher im Transkript gefunden.")
            return {}

        # 2. Interaktions-Klassifikation & Diskurs-Alignment via LLM-Micro-Batching
        main_spks = list(stats["speakers"].keys())
        interaction_styles = self._classify_interactions(
            transcript_data.get("segments", []), main_spks
        )
        for spk, style_data in interaction_styles.items():
            if spk in stats["speakers"]:
                stats["speakers"][spk]["interaction_style"] = style_data

        # 3. LLM Profiling (angereichert mit Interaktionsstilen & Listening-Scores)
        profiles = self.profile_speakers(transcript_data, stats, title=episode_title)

        combined_result = {
            "title": episode_title,
            "statistics": stats,
            "profiles": profiles.get("profiles", {}),
            "chemistry_dynamics": profiles.get("chemistry_dynamics", ""),
        }

        target_dir = output_dir or self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # JSON Report speichern
        json_path = target_dir / f"{file_stem}_speakers.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(combined_result, f, ensure_ascii=False, indent=2)

        # Markdown Report speichern
        md_path = target_dir / f"{file_stem}_speakers.md"
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
        md.append("| **Fragen gestellt** | " + " | ".join(f"{stats['speakers'][s].get('question_count', 0)} ({stats['speakers'][s].get('question_percentage', 0.0)}%)" for s in speakers) + " |")
        md.append("| **Unterbrechungen (erteilt / erlitten)** | " + " | ".join(f"{stats['speakers'][s].get('interruptions_made', 0)}x / {stats['speakers'][s].get('interruptions_suffered', 0)}x" for s in speakers) + " |")

        # Unterbrechungs-Stil
        def fmt_style(s):
            style = stats['speakers'][s].get('interaction_style', {})
            if not style or style.get('evaluated_count', 0) == 0:
                return "-"
            return f"{style['steering_pct']}% *Steer* / {style['supportive_pct']}% *Supp* / {style['disruptive_pct']}% *Disr*"
        md.append("| **Unterbrechungs-Stil (Steer/Supp/Disr)** | " + " | ".join(fmt_style(s) for s in speakers) + " |")

        # Zuhör- & Resonanz-Score
        def fmt_listen(s):
            style = stats['speakers'][s].get('interaction_style', {})
            score = style.get('listening_score', 0.0)
            return f"**{score}** / 5.0" if score > 0 else "-"
        md.append("| **Zuhör- & Resonanz-Score** | " + " | ".join(fmt_listen(s) for s in speakers) + " |")

        md.append("| **Wortschatz-Vielfalt (MTLD Inhalt / Gesamt)** | " + " | ".join(f"**{stats['speakers'][s].get('mtld_content_score', 0)}** ({stats['speakers'][s].get('mtld_score', 0)} gesamt)" for s in speakers) + " |")

        # Top Füllwörter
        def fmt_fillers(s):
            fls = stats['speakers'][s].get('top_fillers', [])
            return ", ".join(f"*{f['word']}* ({f['count']}x)" for f in fls[:4]) if fls else "-"
        md.append("| **Typische Füllwörter** | " + " | ".join(fmt_fillers(s) for s in speakers) + " |")

        # Top Inhaltswörter (Nicht-Füllwörter)
        def fmt_content(s):
            cnt = stats['speakers'][s].get('top_content_words', [])
            return ", ".join(f"`{c['word']}` ({c['count']}x)" for c in cnt[:5]) if cnt else "-"
        md.append("| **Top-Themenbegriffe (Nicht-Füllwörter)** | " + " | ".join(fmt_content(s) for s in speakers) + " |")

        md.append("")
        md.append("> 💡 **Legende:**  ")
        md.append("> - **Wortschatz-Vielfalt (MTLD):** Measure of Textual Lexical Diversity. **Inhalt** filtert Funktionswörter heraus und spiegelt den echten thematischen Wortschatz wider (höher = reichhaltigere Fachbegriffe). **Gesamt** erfasst die Token-Vielfalt inklusive rhetorischer Schleifen und Partikeln der Alltagssprache.  ")
        md.append("> - **Unterbrechungs-Stil:** *Steer* = Themenlenkung & investigatives Nachhaken; *Supp* = Kooperativ (Zustimmung, Lachen, Satz vollenden); *Disr* = Disruptiv (Widerspruch, Redezeit an sich reißen).  ")
        md.append("> - **Zuhör- & Resonanz-Score:** **1.0** (Vorredner ignoriert / harter Themenwechsel) bis **5.0** (geht direkt und tief auf das Vorargument ein).\n")

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
