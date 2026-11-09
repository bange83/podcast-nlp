import json
import re
from dataclasses import dataclass
from pathlib import Path

import feedparser
import requests
from slugify import slugify
from tqdm import tqdm

from .config import AUDIO_DIR, get_podcast_audio_dir

# Standard Browser User-Agent, um 403-Sperren von Podcast-CDNs (z.B. Acast, Megaphone) zu verhindern
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "*/*",
}


@dataclass
class PodcastEpisode:
    title: str
    published: str
    audio_url: str
    description: str
    podcast_title: str = ""
    podcast_author: str = ""
    episode_number: int | None = None
    local_path: Path | None = None


class PodcastDownloader:
    def __init__(self, output_dir: Path = AUDIO_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def search_podcasts(query: str, limit: int = 5) -> list[dict]:
        """Sucht Podcasts über das offene Apple Podcasts / iTunes Verzeichnis."""
        search_url = "https://itunes.apple.com/search"
        params = {
            "term": query,
            "media": "podcast",
            "entity": "podcast",
            "limit": limit,
        }
        try:
            resp = requests.get(search_url, params=params, headers=DEFAULT_HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", []):
                feed_url = item.get("feedUrl")
                if feed_url:
                    results.append(
                        {
                            "name": item.get("collectionName"),
                            "artist": item.get("artistName"),
                            "feed_url": feed_url,
                            "track_count": item.get("trackCount", 0),
                        }
                    )
            return results
        except Exception as e:
            print(f"[Downloader] Fehler bei der Podcast-Suche: {e}")
            return []

    def fetch_feed_episodes(
        self, feed_url: str, limit: int = 5
    ) -> list[PodcastEpisode]:
        """Liest einen RSS-Feed ein und gibt Metadaten der neuesten Episoden zurück."""
        feed = feedparser.parse(feed_url, agent=DEFAULT_HEADERS["User-Agent"])
        if not feed.entries:
            raise ValueError(f"Keine Episoden im RSS-Feed gefunden: {feed_url}")

        podcast_title = feed.feed.get("title", "")
        podcast_author = feed.feed.get("author", "") or feed.feed.get("itunes_author", "")

        episodes: list[PodcastEpisode] = []
        total_entries = len(feed.entries)
        for idx, entry in enumerate(feed.entries[:limit]):
            audio_url = None
            # Enclosures durchsuchen (standardmäßiger Podcast-Audiolink)
            for enclosure in getattr(entry, "enclosures", []):
                if enclosure.get("type", "").startswith("audio/") or enclosure.get(
                    "href", ""
                ).endswith((".mp3", ".m4a", ".wav", ".aac")):
                    audio_url = enclosure.get("href")
                    break

            # Fallback: Links durchsuchen
            if not audio_url:
                for link in getattr(entry, "links", []):
                    if link.get("type", "").startswith("audio/") or link.get(
                        "href", ""
                    ).endswith((".mp3", ".m4a", ".wav", ".aac")):
                        audio_url = link.get("href")
                        break

            if audio_url:
                title = getattr(entry, "title", "Unbekannte Episode")
                published = getattr(entry, "published", "")
                description = getattr(entry, "summary", "")

                # Folgennummer ermitteln
                ep_num: int | None = None
                raw_ep = getattr(entry, "itunes_episode", None) or getattr(entry, "episode", None)
                if raw_ep is not None:
                    try:
                        ep_num = int(raw_ep)
                    except (ValueError, TypeError):
                        pass

                # Fallback: Aus Titel extrahieren (z.B. "#42", "Folge 42", "Ep. 42", "42 - ...")
                if ep_num is None:
                    match = re.search(r"(?:Folge|Episode|Ep\.?|#)\s*(\d+)", title, re.IGNORECASE)
                    if not match:
                        match = re.match(r"^(\d+)\b", title)
                    if match:
                        try:
                            ep_num = int(match.group(1))
                        except (ValueError, TypeError):
                            pass

                # Fallback: Chronologische Nummerierung aus RSS-Feed-Reihenfolge (neueste oben)
                if ep_num is None and total_entries > 0:
                    ep_num = total_entries - idx

                episodes.append(
                    PodcastEpisode(
                        title=title,
                        published=published,
                        audio_url=audio_url,
                        description=description,
                        podcast_title=podcast_title,
                        podcast_author=podcast_author,
                        episode_number=ep_num,
                    )
                )

        return episodes

    def download_episode(self, episode: PodcastEpisode) -> Path:
        """Lädt eine einzelne Episode mit Fortschrittsbalken in data/<podcast>/audio/ herunter."""
        clean_title = slugify(episode.title, max_length=60) or "episode"
        if episode.episode_number is not None:
            filename = f"{episode.episode_number:04d}_{clean_title}"
        else:
            filename = clean_title

        ext = ".mp3"
        if ".m4a" in episode.audio_url:
            ext = ".m4a"

        # Zielordner: data/<podcast-slug>/audio/ (oder übergebener output_dir als Fallback)
        if episode.podcast_title:
            target_dir = get_podcast_audio_dir(episode.podcast_title)
        else:
            target_dir = self.output_dir
            target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / f"{filename}{ext}"
        meta_file = target_dir / f"{filename}.meta.json"

        if not meta_file.exists():
            meta_data = {
                "title": episode.title,
                "episode_number": episode.episode_number,
                "published": episode.published,
                "audio_url": episode.audio_url,
                "description": episode.description,
                "podcast_title": episode.podcast_title,
                "podcast_author": episode.podcast_author,
            }
            try:
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta_data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        if target_file.exists() and target_file.stat().st_size > 0:
            print(f"[Downloader] Bereits vorhanden: {target_file.relative_to(target_file.parent.parent.parent)}")
            episode.local_path = target_file
            return target_file

        print(f"[Downloader] Starte Download: {episode.title} -> {target_file.name}")
        response = requests.get(episode.audio_url, headers=DEFAULT_HEADERS, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(target_file, "wb") as f,
            tqdm(
                desc=filename[:30],
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        episode.local_path = target_file
        return target_file
