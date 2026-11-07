import json
from dataclasses import dataclass
from pathlib import Path

import feedparser
import requests
from slugify import slugify
from tqdm import tqdm

from .config import AUDIO_DIR


@dataclass
class PodcastEpisode:
    title: str
    published: str
    audio_url: str
    description: str
    podcast_title: str = ""
    podcast_author: str = ""
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
            resp = requests.get(search_url, params=params, timeout=10)
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
        feed = feedparser.parse(feed_url)
        if not feed.entries:
            raise ValueError(f"Keine Episoden im RSS-Feed gefunden: {feed_url}")

        podcast_title = feed.feed.get("title", "")
        podcast_author = feed.feed.get("author", "") or feed.feed.get("itunes_author", "")

        episodes: list[PodcastEpisode] = []
        for entry in feed.entries[:limit]:
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
                episodes.append(
                    PodcastEpisode(
                        title=title,
                        published=published,
                        audio_url=audio_url,
                        description=description,
                        podcast_title=podcast_title,
                        podcast_author=podcast_author,
                    )
                )

        return episodes

    def download_episode(self, episode: PodcastEpisode) -> Path:
        """Lädt eine einzelne Episode mit Fortschrittsbalken herunter."""
        clean_title = slugify(episode.title, max_length=60) or "episode"
        ext = ".mp3"
        if ".m4a" in episode.audio_url:
            ext = ".m4a"

        target_file = self.output_dir / f"{clean_title}{ext}"

        meta_file = self.output_dir / f"{clean_title}.meta.json"
        if not meta_file.exists():
            meta_data = {
                "title": episode.title,
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
            print(f"[Downloader] Bereits vorhanden: {target_file.name}")
            episode.local_path = target_file
            return target_file

        print(f"[Downloader] Starte Download: {episode.title}")
        response = requests.get(episode.audio_url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(target_file, "wb") as f,
            tqdm(
                desc=clean_title[:30],
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
