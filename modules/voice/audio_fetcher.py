"""
Audio Fetcher   Earnings Call Downloader

Uses YouTube Data API to search for CEO earnings calls, then
yt-dlp to download the audio track for analysis.
"""

import re
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

from googleapiclient.discovery import build

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import YOUTUBE_API_KEY, DOWNLOADS_DIR


class AudioFetcher:
    """
    Search YouTube for earnings calls and download audio.

    Usage:
        fetcher = AudioFetcher()
        path = fetcher.fetch_earnings_call("Reliance Industries", "Q3 2024")
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or YOUTUBE_API_KEY
        if not self.api_key:
            raise ValueError("YouTube API key is required. Set YOUTUBE_API_KEY in .env")
        self.youtube = build("youtube", "v3", developerKey=self.api_key)
        self.save_dir = DOWNLOADS_DIR / "audio"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    #    Public API                                                           

    def search_earnings_calls(
        self,
        company_name: str,
        quarter: str = "",
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search YouTube for earnings call videos.

        Args:
            company_name: e.g. "Infosys"
            quarter: e.g. "Q3 2024" (optional, broadens search if empty)
            max_results: max videos to return

        Returns list of dicts: {video_id, title, channel, published_at, url}
        """
        query = f"{company_name} earnings call {quarter} CEO".strip()

        request = self.youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=max_results,
            order="relevance",
            videoDuration="long",        # earnings calls are usually 30+ min
        )
        response = request.execute()

        results = []
        for item in response.get("items", []):
            vid_id = item["id"]["videoId"]
            snippet = item["snippet"]
            results.append({
                "video_id": vid_id,
                "title": snippet["title"],
                "channel": snippet["channelTitle"],
                "published_at": snippet["publishedAt"],
                "url": f"https://www.youtube.com/watch?v={vid_id}",
            })
        return results

    def download_audio(
        self,
        video_url: str,
        filename: str = None,
    ) -> Optional[Path]:
        """
        Download audio from a YouTube video using yt-dlp.

        Returns path to the downloaded .wav file, or None on failure.
        """
        if filename is None:
            # Extract video ID for filename
            match = re.search(r"v=([A-Za-z0-9_-]+)", video_url)
            filename = match.group(1) if match else "audio"

        output_path = self.save_dir / f"{filename}.wav"
        if output_path.exists():
            print(f"    Audio already exists: {output_path.name}")
            return output_path

        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--output", str(self.save_dir / f"{filename}.%(ext)s"),
            "--no-playlist",
            "--quiet",
            video_url,
        ]

        try:
            print(f"    Downloading audio from: {video_url}")
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            if output_path.exists():
                print(f"    Saved: {output_path.name}")
                return output_path
            # yt-dlp may save with different extension; check for any match
            for f in self.save_dir.glob(f"{filename}.*"):
                if f.suffix in (".wav", ".mp3", ".m4a", ".opus"):
                    print(f"    Saved: {f.name}")
                    return f
            print("    Download succeeded but file not found")
            return None
        except subprocess.CalledProcessError as e:
            print(f"    yt-dlp error: {e.stderr[:200] if e.stderr else str(e)}")
            return None
        except FileNotFoundError:
            print("    yt-dlp not found. Install with: pip install yt-dlp")
            return None

    def fetch_earnings_call(
        self,
        company_name: str,
        quarter: str = "",
    ) -> Optional[Path]:
        """
        One-shot: Search for the most relevant earnings call and download audio.
        """
        results = self.search_earnings_calls(company_name, quarter, max_results=3)
        if not results:
            print(f"    No earnings call videos found for '{company_name} {quarter}'")
            return None

        # Pick the first (most relevant) result
        best = results[0]
        print(f"    Best match: {best['title']}")
        safe_name = re.sub(r"[^\w\-]", "_", f"{company_name}_{quarter}".strip("_"))
        return self.download_audio(best["url"], filename=safe_name)


#    Quick test                                                               
if __name__ == "__main__":
    fetcher = AudioFetcher()
    results = fetcher.search_earnings_calls("Infosys", "Q3 2024")
    print(f"\nFound {len(results)} results:")
    for r in results:
        print(f"    {r['title']}")
        print(f"    {r['url']}")
