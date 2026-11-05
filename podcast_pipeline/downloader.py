import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import feedparser
import requests
from tqdm import tqdm
from slugify import slugify

from .config import AUDIO_DIR


@dataclass
class PodcastEpisode:
    title: str
    published: str
    audio_url: str
    description: str
    local_path: Optional[Path] = None


class PodcastDownloader:
    def __init__(self, output_dir: Path = AUDIO_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_feed_episodes(self, feed_url: str, limit: int = 5) -> List[PodcastEpisode]:
        """Liest einen RSS-Feed ein und gibt Metadaten der neuesten Episoden zurück."""
        feed = feedparser.parse(feed_url)
        if not feed.entries:
            raise ValueError(f"Keine Episoden im RSS-Feed gefunden: {feed_url}")

        episodes: List[PodcastEpisode] = []
        for entry in feed.entries[:limit]:
            audio_url = None
            # Enclosures durchsuchen (standardmäßiger Podcast-Audiolink)
            for enclosure in getattr(entry, "enclosures", []):
                if enclosure.get("type", "").startswith("audio/") or enclosure.get("href", "").endswith((".mp3", ".m4a", ".wav", ".aac")):
                    audio_url = enclosure.get("href")
                    break

            # Fallback: Links durchsuchen
            if not audio_url:
                for link in getattr(entry, "links", []):
                    if link.get("type", "").startswith("audio/") or link.get("href", "").endswith((".mp3", ".m4a", ".wav", ".aac")):
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
                        description=description
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

        if target_file.exists() and target_file.stat().st_size > 0:
            print(f"[Downloader] Bereits vorhanden: {target_file.name}")
            episode.local_path = target_file
            return target_file

        print(f"[Downloader] Starte Download: {episode.title}")
        response = requests.get(episode.audio_url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(target_file, "wb") as f, tqdm(
            desc=clean_title[:30],
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        episode.local_path = target_file
        return target_file
