import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import ollama

from .config import REPORTS_DIR, DEFAULT_OLLAMA_MODEL


class LocalNLPAnalyzer:
    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        output_dir: Path = REPORTS_DIR,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def map_speaker_names(
        self,
        transcript_data: Dict[str, Any],
        candidate_hosts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Versucht via lokalem LLM generische Sprecher-IDs (SPEAKER_07 etc.) echten Namen zuzuordnen."""
        segments = transcript_data.get("segments", [])

        # Häufigkeiten der Sprecher ermitteln
        counts = {}
        for s in segments:
            spk = s.get("speaker")
            if spk and spk != "UNKNOWN":
                counts[spk] = counts.get(spk, 0) + 1

        if len(counts) < 2:
            return transcript_data

        # Die 2 dominantesten Sprecher finden (z.B. Jan & Olli)
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

        hosts_hint = f"Mögliche Personen: {', '.join(candidate_hosts)}" if candidate_hosts else "Z.B. Jan Böhmermann, Olli Schulz bei Fest & Flauschig."

        prompt = f"""Hier ist ein Dialog aus einem Podcast zwischen den beiden Hauptsprechern ({', '.join(top_speakers)}):

{sample_text}

Aufgabe: Ordne den beiden Sprecher-IDs ({', '.join(top_speakers)}) die echten Namen der Personen zu ({hosts_hint}), basierend darauf wie sie sprechen und worüber sie reden.

Antworte ausschließlich im JSON-Format:
{{
  "{top_speakers[0]}": "Name der Person",
  "{top_speakers[1]}": "Name der Person"
}}
"""
        try:
            resp = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1}
            )
            raw = resp["message"]["content"].strip()
            if "{" in raw and "}" in raw:
                raw_json = raw[raw.find("{"):raw.rfind("}")+1]
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

    def analyze_transcript(
        self,
        transcript_data: Dict[str, Any],
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """Führt umfassende NLP-Analysen auf dem Transkript mittels lokalem LLM durch."""
        text = transcript_data.get("full_text", "")
        file_stem = Path(transcript_data.get("audio_file", "podcast")).stem
        episode_title = title or file_stem

        print(f"[NLPAnalyzer] Starte Analyse mit lokalem LLM '{self.model_name}'...")

        # System Prompt für präzise, strukturierte Analyse
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
    "persons": ["Tatsächlich im Podcast genannte reale Personen (z.B. Jan Böhmermann, Olli Schulz, Gäste)"],
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
                options={"temperature": 0.3}
            )

            raw_content = response["message"]["content"].strip()

            # Bereinigen falls das Modell Markdown-Codeblöcke mitliefert
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
                    "raw_analysis": raw_content
                }

        except Exception as e:
            print(f"[NLPAnalyzer] Fehler bei der Ollama-Anfrage: {e}")
            raise

        # Speichern als JSON
        json_report_path = self.output_dir / f"{file_stem}_analysis.json"
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)

        # Zusätzlich als schöner Markdown-Report speichern
        md_report_path = self.output_dir / f"{file_stem}_report.md"
        self._save_markdown_report(analysis_result, md_report_path)

        print(f"[NLPAnalyzer] Analyse abgeschlossen! Gespeichert unter:\n  -> {json_report_path}\n  -> {md_report_path}")
        return analysis_result

    def _save_markdown_report(self, data: Dict[str, Any], path: Path) -> None:
        """Erzeugt einen ansprechenden Markdown-Bericht aus den Analyseergebnissen."""
        md = []
        md.append(f"# Podcast-Analyse: {data.get('title', 'Episode')}\n")

        if "one_sentence_summary" in data:
            md.append(f"> **TL;DR:** {data['one_sentence_summary']}\n")

        if "detailed_summary" in data:
            md.append("## 📝 Zusammenfassung")
            md.append(data["detailed_summary"] + "\n")

        if "key_takeaways" in data and data["key_takeaways"]:
            md.append("## 💡 Wichtigste Kernaussagen")
            for item in data["key_takeaways"]:
                md.append(f"- {item}")
            md.append("")

        if "topics" in data and data["topics"]:
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

        if "actionable_insights" in data and data["actionable_insights"]:
            md.append("## 🚀 Erkenntnisse & Praxistipps")
            for item in data["actionable_insights"]:
                md.append(f"- {item}")
            md.append("")

        if "raw_analysis" in data:
            md.append("## Analyseergebnis (Rohdaten)")
            md.append(data["raw_analysis"])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
