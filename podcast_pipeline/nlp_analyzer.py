import json
from pathlib import Path
from typing import Any

import ollama

from .config import DEFAULT_OLLAMA_MODEL, REPORTS_DIR


class LocalNLPAnalyzer:
    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        output_dir: Path = REPORTS_DIR,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Formatiert Sekunden in MM:SS oder HH:MM:SS."""
        secs = int(seconds)
        mins = secs // 60
        rem_secs = secs % 60
        hours = mins // 60
        rem_mins = mins % 60
        if hours > 0:
            return f"{hours:02d}:{rem_mins:02d}:{rem_secs:02d}"
        return f"{rem_mins:02d}:{rem_secs:02d}"

    @staticmethod
    def _parse_timestamp(val: Any) -> float:
        """Parst 'MM:SS' oder float Sekunden."""
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val).strip()
        parts = val_str.split(":")
        try:
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(val_str)
        except ValueError:
            return 0.0

    def map_speaker_names(
        self,
        transcript_data: dict[str, Any],
        podcast_title: str | None = None,
        podcast_author: str | None = None,
        episode_title: str | None = None,
        episode_description: str | None = None,
        candidate_hosts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Versucht via lokalem LLM generische Sprecher-IDs (SPEAKER_00 etc.) echten Namen zuzuordnen."""
        segments = transcript_data.get("segments", [])

        # Häufigkeiten der Sprecher ermitteln
        counts: dict[str, int] = {}
        for s in segments:
            spk = s.get("speaker")
            if spk and spk != "UNKNOWN":
                counts[spk] = counts.get(spk, 0) + 1

        if len(counts) < 2:
            return transcript_data

        # Die 2 dominantesten Sprecher finden
        top_speakers = [spk for spk, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:2]]

        # Ausschnitt finden, in dem diese beiden tatsächlich miteinander reden (Werbung überspringen)
        sample_dialogue = []
        for s in segments:
            spk = s.get("speaker")
            if spk in top_speakers:
                sample_dialogue.append(f"{spk}: {s.get('text')}")
            if len(sample_dialogue) >= 40:
                break

        sample_text = "\n".join(sample_dialogue)

        context_lines = []
        if podcast_title:
            context_lines.append(f"- Podcast: {podcast_title}")
        if podcast_author:
            context_lines.append(f"- Moderation / Host(s) / Autor: {podcast_author}")
        if episode_title:
            context_lines.append(f"- Episode: {episode_title}")
        if episode_description:
            clean_desc = episode_description.strip()[:300].replace("\n", " ")
            context_lines.append(f"- Episoden-Beschreibung: {clean_desc}")
        if candidate_hosts:
            context_lines.append(f"- Mögliche Personen: {', '.join(candidate_hosts)}")

        meta_info = "\n".join(context_lines) if context_lines else "- Kein spezifischer Podcast-Kontext verfügbar"

        prompt = f"""Du bist ein Experte für Podcast-Analyse und Sprecherzuordnung.

Hier sind die Metadaten der Podcast-Episode:
{meta_info}

Hier ist ein repräsentativer Gesprächsausschnitt zwischen den beiden Hauptsprechern ({', '.join(top_speakers)}):

{sample_text}

Aufgabe:
Ordne den beiden Sprecher-IDs ({', '.join(top_speakers)}) die echten Namen der jeweiligen Personen zu.
Kriterien:
1. Nutze die Metadaten (Host-Namen, Autoren, Titel) und den Gesprächsverlauf (wie sprechen sich die Personen an, welche Rollen nehmen sie ein?).
2. Wenn sich ein Name klar zuordnen lässt, verwende diesen Namen.
3. Falls ein Sprecher nicht zweifelsfrei identifiziert werden kann, verwende einen neutralen Bezeichner wie "Host 1" / "Host 2" oder behalte die Sprecher-ID. Erfinde keinesfalls Personen.

Antworte ausschließlich im folgenden JSON-Format ohne Markdown-Codeblöcke:
{{
  "{top_speakers[0]}": "Name der Person",
  "{top_speakers[1]}": "Name der Person"
}}
"""
        try:
            resp = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
            )
            raw = resp["message"]["content"].strip()
            if "{" in raw and "}" in raw:
                raw_json = raw[raw.find("{") : raw.rfind("}") + 1]
                mapping = json.loads(raw_json)
                if mapping:
                    print(f"[NLPAnalyzer] Erkannte Sprecher-Zuordnung: {mapping}")
                    new_segments = []
                    new_text_lines = []
                    for s in segments:
                        orig_spk = s.get("speaker")
                        real_name = mapping.get(orig_spk, orig_spk)
                        s["speaker"] = real_name
                        new_segments.append(s)
                        if real_name != "UNKNOWN":
                            new_text_lines.append(f"[{s['start']}s - {s['end']}s] {real_name}: {s['text']}")
                        else:
                            new_text_lines.append(f"[{s['start']}s - {s['end']}s] {s['text']}")

                    transcript_data["segments"] = new_segments
                    transcript_data["full_text"] = "\n".join(new_text_lines)
                    transcript_data["speaker_mapping"] = mapping
        except Exception as e:
            print(f"[NLPAnalyzer] Konnte Sprecher nicht automatisch auflösen: {e}")

        return transcript_data

    # =========================================================================
    # Stufe 1: Themen-Segmentierung (Kapitel & Zeitstempel)
    # =========================================================================

    def segment_topics(
        self,
        transcript_data: dict[str, Any],
        title: str | None = None,
    ) -> list[dict[str, Any]]:
        """Erkennt automatisch die Gesprächsthemen und teilt den Podcast in Kapitelblöcke ein."""
        segments = transcript_data.get("segments", [])
        if not segments:
            return []

        # Komprimierte Zeitleiste erzeugen (z.B. alle ~60-90 Sekunden)
        timeline_entries = []
        last_t = -100.0
        for s in segments:
            start_t = float(s.get("start", 0))
            if start_t - last_t >= 60.0:
                spk = s.get("speaker", "Sprecher")
                text_snippet = s.get("text", "").strip()[:100]
                ts_str = self._format_timestamp(start_t)
                timeline_entries.append(f"[{ts_str}] {spk}: {text_snippet}")
                last_t = start_t

        timeline_text = "\n".join(timeline_entries)
        episode_title = title or transcript_data.get("audio_file", "Podcast Episode")

        prompt = f"""Du bist ein erfahrener Podcast-Redakteur.
Hier ist die chronologische Zeitleiste des Podcasts '{episode_title}' mit Zeitstempeln (im Format MM:SS):

{timeline_text}

Aufgabe:
Erkenne die verschiedenen Gesprächsthemen und teile den Podcast in klare, inhaltliche Kapitel (Themenblöcke) ein.
Für jedes Thema:
- Startzeitpunkt ("MM:SS")
- Endzeitpunkt ("MM:SS")
- Ein prägnanter Thementitel
- 1-2 Sätze Kurzbeschreibung
- 3-5 konkrete Schlüsselwörter / Entitäten (reale Personen, Schiffe, Orte, Marken, Begriffe)

Antworte ausschließlich im folgenden JSON-Format ohne Markdown-Codeblöcke:
{{
  "chapters": [
    {{
      "start": "00:00",
      "end": "12:30",
      "title": "Intro & Begrüßung",
      "description": "Geplänkel der Moderatoren über die Woche...",
      "keywords": ["Begrüßung", "Wetter", "Wochenende"]
    }}
  ]
}}
"""
        print(f"[NLPAnalyzer] Erstelle thematische Kapitelübersicht mit '{self.model_name}'...")
        try:
            resp = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
            )
            raw = resp["message"]["content"].strip()
            if "{" in raw and "}" in raw:
                raw_json = raw[raw.find("{") : raw.rfind("}") + 1]
                data = json.loads(raw_json)
                chapters = data.get("chapters", [])
                for c in chapters:
                    c["start_seconds"] = self._parse_timestamp(c.get("start", 0))
                    c["end_seconds"] = self._parse_timestamp(c.get("end", 0))
                if chapters:
                    print(f"[NLPAnalyzer] {len(chapters)} Kapitel/Themenblöcke identifiziert!")
                    return chapters
        except Exception as e:
            print(f"[NLPAnalyzer] Fehler bei Themen-Segmentierung: {e}")

        return []

    # =========================================================================
    # Stufe 2: Kontext-verankertes Refinement (Korrektur pro Thema)
    # =========================================================================

    def refine_topic_segments(
        self,
        transcript_data: dict[str, Any],
        chapters: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Korrigiert phonetische Hörfehler von Whisper gezielt über eine sichere Find-and-Replace-Liste."""
        segments = transcript_data.get("segments", [])
        if not segments or not chapters:
            return transcript_data

        print(f"[NLPAnalyzer] Starte kontextuelles Refinement für {len(chapters)} Themenblöcke...")
        corrections_count = 0

        for ch_idx, ch in enumerate(chapters, 1):
            ch_start = ch.get("start_seconds", 0.0)
            ch_end = ch.get("end_seconds", float("inf"))
            ch_title = ch.get("title", f"Thema {ch_idx}")
            ch_keywords = ", ".join(ch.get("keywords", []))

            # Segmente dieses Kapitels filtern
            matching_segments = [
                s for s in segments
                if ch_start <= float(s.get("start", 0)) <= ch_end
            ]

            if not matching_segments:
                continue

            # In Portionen von ca. 35 Segmenten verarbeiten
            chunk_size = 35
            for chunk_offset in range(0, len(matching_segments), chunk_size):
                sub_segments = matching_segments[chunk_offset : chunk_offset + chunk_size]
                chunk_dialogue = [
                    f"{s.get('speaker', 'Sprecher')}: {s.get('text', '')}"
                    for s in sub_segments
                ]
                dialogue_text = "\n".join(chunk_dialogue)

                prompt = f"""Du bist ein professioneller Audio-Transkript-Lektor.
Thema des aktuellen Abschnitts: "{ch_title}"
Schlüsselbegriffe & Kontext: {ch_keywords}

Hier ist der gesprochene Dialog-Ausschnitt:
---
{dialogue_text}
---

Aufgabe:
Finde offensichtliche akustische Transkriptions- und Hörfehler von Whisper (besonders Eigennamen, Markennamen, Schiffe, Bands, Personen, Orte, Fachbegriffe oder Redewendungen, die akustisch missverstanden wurden).

Beispiele:
- "Mind-Shift 3" -> "Mein Schiff 3" (bei Thema Schifffahrt/Kreuzfahrt)
- "Bocker" -> "Booker" (bei Thema Musik/Tournee)
- "Kimmich" -> "Chemie"

WICHTIG:
- Erstelle eine Liste von konkreten Wort-Ersetzungen ("original" -> "corrected").
- Verändere NIEMALS grammatikalische Eigenheiten, Füllwörter, Satzbau oder Umgangssprache.
- Wenn im Text keine offensichtlichen Hörfehler vorliegen, gib ein leeres Array [] zurück.

Antworte ausschließlich im folgenden JSON-Format ohne Markdown-Codeblöcke:
{{
  "corrections": [
    {{
      "original": "falscher Begriff wie er im Text steht",
      "corrected": "korrekter Begriff"
    }}
  ]
}}
"""
                try:
                    resp = ollama.chat(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": 0.1},
                    )
                    raw_content = resp["message"]["content"].strip()
                    if "{" in raw_content and "}" in raw_content:
                        raw_json = raw_content[raw_content.find("{") : raw_content.rfind("}") + 1]
                        data = json.loads(raw_json)
                        corrections = data.get("corrections", [])
                        for c in corrections:
                            orig = c.get("original", "").strip()
                            repl = c.get("corrected", "").strip()
                            if not orig or not repl or orig.lower() == repl.lower():
                                continue

                            # Ersetze gezielt in den Segmenten dieses Chunks
                            replaced_any = False
                            for s in sub_segments:
                                text_val = s.get("text", "")
                                if orig in text_val:
                                    s["text"] = text_val.replace(orig, repl)
                                    replaced_any = True
                            if replaced_any:
                                print(f"  ✨ [Refiner] '{orig}' -> '{repl}'")
                                corrections_count += 1
                except Exception as e:
                    print(f"[NLPAnalyzer] Fehler beim Refinement eines Chunks: {e}")

        print(f"[NLPAnalyzer] Refinement abgeschlossen ({corrections_count} gezielte Begriff-Korrektur(en)).")

        # Volltext neu aufbauen
        new_text_lines = []
        for s in segments:
            spk = s.get("speaker")
            if spk and spk != "UNKNOWN":
                new_text_lines.append(f"[{s['start']}s - {s['end']}s] {spk}: {s['text']}")
            else:
                new_text_lines.append(f"[{s['start']}s - {s['end']}s] {s['text']}")

        transcript_data["segments"] = segments
        transcript_data["full_text"] = "\n".join(new_text_lines)
        transcript_data["chapters"] = chapters
        return transcript_data

    # =========================================================================
    # Stufe 3: Umfassende Synthese & Analyse
    # =========================================================================

    def analyze_transcript(
        self,
        transcript_data: dict[str, Any],
        title: str | None = None,
        refine_chapters: bool = True,
    ) -> dict[str, Any]:
        """Führt Themen-Segmentierung, kontextuelles Refinement und die finale NLP-Analyse durch."""
        file_stem = Path(transcript_data.get("audio_file", "podcast")).stem
        episode_title = title or file_stem

        # 1. Themen-Segmentierung & Kapitelmarker
        chapters = transcript_data.get("chapters")
        if not chapters:
            chapters = self.segment_topics(transcript_data, title=episode_title)

        # 2. Kontextuelles Refinement pro Kapitel
        if refine_chapters and chapters:
            transcript_data = self.refine_topic_segments(transcript_data, chapters)

        text = transcript_data.get("full_text", "")

        print(f"[NLPAnalyzer] Starte finale Synthese mit lokalem LLM '{self.model_name}'...")

        # System Prompt für präzise Analyse
        system_prompt = (
            "Du bist ein fortschrittlicher NLP- und Podcast-Analyst. "
            "Deine Aufgabe ist es, gesprochene Podcast-Transkripte zu analysieren und "
            "eine strukturierte, präzise Zusammenfassung mit Kernerkenntnissen zu liefern. "
            "Antworte immer auf Deutsch, es sei denn, das Transkript erfordert zwingend eine andere Sprache."
        )

        user_prompt = f"""Hier ist das Transkript einer Podcast-Folge:
Titel: {episode_title}

--- TRANSKRIPT ANFANG ---
{text[:24000]}  # Begrenzung für Standard-Kontextfenster
--- TRANSKRIPT ENDE ---

Erstelle eine fundierte Analyse und antworte im folgenden JSON-Format:
{{
  "title": "{episode_title}",
  "one_sentence_summary": "Eine prägnante einzeilige Kernaussage der Folge",
  "detailed_summary": "Ausführliche Zusammenfassung der Hauptargumente und Diskussionen (3-5 Absätze)",
  "key_takeaways": [
    "Kernaussage 1",
    "Kernaussage 2",
    "Kernaussage 3"
  ],
  "topics": ["Konkretes Thema 1", "Konkretes Thema 2"],
  "named_entities": {{
    "persons": ["Tatsächlich im Podcast genannte reale Personen, Moderatoren, Gäste"],
    "organizations": ["Erwähnte Institutionen, Sender, Unternehmen"],
    "tools_and_media": ["Erwähnte Bücher, Filme, Songs, Software"]
  }},
  "sentiment_and_tone": {{
    "overall_sentiment": "positiv | neutral | kritisch | gemischt",
    "tone": "z.B. analytisch, humorvoll, satirisch, kontrovers, nachdenklich",
    "explanation": "Kurze Begründung der Stimmung"
  }},
  "actionable_insights": [
    "Konkreter Praxistipp oder Denkanstoß 1",
    "Konkreter Praxistipp 2"
  ]
}}

WICHTIG:
- Erfinde keine Platzhalter wie 'Person 1' oder 'Unternehmen 1'.
- Nenne nur Entitäten, die tatsächlich im Transkript vorkommen. Wenn nichts genannt wurde, gib ein leeres Array [] zurück.
- Antworte ausschließlich mit dem gültigen JSON-Objekt, ohne Markdown-Backticks drumherum.
"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.3},
            )

            raw_content = response["message"]["content"].strip()

            if raw_content.startswith("```"):
                lines = raw_content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_content = "\n".join(lines).strip()

            try:
                analysis_result = json.loads(raw_content)
            except json.JSONDecodeError:
                analysis_result = {
                    "title": episode_title,
                    "raw_analysis": raw_content,
                }

        except Exception as e:
            print(f"[NLPAnalyzer] Fehler bei der Ollama-Anfrage: {e}")
            raise

        if chapters:
            analysis_result["chapters"] = chapters

        # Speichern als JSON
        json_report_path = self.output_dir / f"{file_stem}_analysis.json"
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)

        # Zusätzlich als schöner Markdown-Report speichern
        md_report_path = self.output_dir / f"{file_stem}_report.md"
        self._save_markdown_report(analysis_result, md_report_path)

        print(f"[NLPAnalyzer] Analyse abgeschlossen! Gespeichert unter:\n  -> {json_report_path}\n  -> {md_report_path}")
        return analysis_result

    def _save_markdown_report(self, data: dict[str, Any], path: Path) -> None:
        """Erzeugt einen ansprechenden Markdown-Bericht aus den Analyseergebnissen."""
        md = []
        md.append(f"# Podcast-Analyse: {data.get('title', 'Episode')}\n")

        if "one_sentence_summary" in data:
            md.append(f"> **TL;DR:** {data['one_sentence_summary']}\n")

        # ⏱️ Kapitelübersicht falls vorhanden
        if data.get("chapters"):
            md.append("## ⏱️ Themen- & Kapitelübersicht\n")
            md.append("| Zeitstempel | Thema | Beschreibung |")
            md.append("|---|---|---|")
            for c in data["chapters"]:
                start_ts = c.get("start", "00:00")
                end_ts = c.get("end", "")
                ts_label = f"`{start_ts} - {end_ts}`" if end_ts else f"`{start_ts}`"
                t_title = c.get("title", "Thema").replace("|", "-")
                t_desc = c.get("description", "").replace("|", "-")
                md.append(f"| {ts_label} | **{t_title}** | {t_desc} |")
            md.append("")

        if "detailed_summary" in data:
            md.append("## 📝 Zusammenfassung")
            md.append(data["detailed_summary"] + "\n")

        if data.get("key_takeaways"):
            md.append("## 💡 Wichtigste Kernaussagen")
            for item in data["key_takeaways"]:
                md.append(f"- {item}")
            md.append("")

        if data.get("topics"):
            md.append("## 🏷️ Themen")
            md.append(", ".join(f"`{t}`" for t in data["topics"]) + "\n")

        if "named_entities" in data and isinstance(data["named_entities"], dict):
            md.append("## 🔍 Erwähnte Entitäten")
            for category, items in data["named_entities"].items():
                if items:
                    cat_name = category.replace("_", " ").title()
                    md.append(f"- **{cat_name}:** {', '.join(items)}")
            md.append("")

        if "sentiment_and_tone" in data and isinstance(data["sentiment_and_tone"], dict):
            st = data["sentiment_and_tone"]
            md.append("## 📊 Tonalität & Sentiment")
            md.append(f"- **Stimmung:** {st.get('overall_sentiment', '-')}")
            md.append(f"- **Tonfall:** {st.get('tone', '-')}")
            if "explanation" in st:
                md.append(f"- **Begründung:** {st['explanation']}")
            md.append("")

        if data.get("actionable_insights"):
            md.append("## 🚀 Erkenntnisse & Praxistipps")
            for item in data["actionable_insights"]:
                md.append(f"- {item}")
            md.append("")

        if "raw_analysis" in data:
            md.append("## Analyseergebnis (Rohdaten)")
            md.append(data["raw_analysis"])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
