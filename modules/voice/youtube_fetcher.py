
import os
import yt_dlp
from pathlib import Path

class YouTubeFetcher:
    """
    Downloads audio from YouTube videos (Earnigns Calls).
    """
    def __init__(self, save_dir: str = "downloads/audio"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def download_audio(self, url: str):
        """
        Downloads best audio from YouTube URL and converts to WAV.
        Returns path to the downloaded file.
        """
        import imageio_ffmpeg
        import shutil
        import tempfile
        
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        # yt-dlp expects 'ffmpeg.exe' in PATH for splitting, but imageio has versioned name.
        # Create a temp dir, copy it as ffmpeg.exe, and add to PATH.
        temp_ffmpeg_dir = Path(tempfile.gettempdir()) / "ffmpeg_bin"
        temp_ffmpeg_dir.mkdir(exist_ok=True)
        target_ffmpeg = temp_ffmpeg_dir / "ffmpeg.exe"
        
        # Only copy if not exists or size differs (simple cache)
        if not target_ffmpeg.exists():
            shutil.copy(ffmpeg_exe, target_ffmpeg)
            
        # Add to PATH
        os.environ["PATH"] = str(temp_ffmpeg_dir) + os.pathsep + os.environ["PATH"]
        
        print(f"DEBUG: Setup FFmpeg shim at {target_ffmpeg}")

        if not url or "youtube.com" not in url and "youtu.be" not in url:
            return None

        # Clean filename template
        out_tmpl = str(self.save_dir / '%(title)s.%(ext)s')

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': out_tmpl,
            'ffmpeg_location': str(target_ffmpeg), # specific path
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            # OPTIMIZATION: Only download first 5 minutes (300s) to avoid timeouts on long calls
            'download_ranges': lambda info, ydl: [{'start_time': 0, 'end_time': 300}],
            'noplaylist': True,
            'quiet': True
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'audio')
                # yt-dlp sanitizes filename, so we need to find it
                # simpler approach: return the first wav file matching the pattern?
                # Actually, prepare_filename gives the output filename before conversion.
                
                # Robust way: Search for the created file in save_dir
                # Because we don't know exactly how it sanitized the title.
                # Or just return the path predicted by prepare_filename, but with .wav ext.
                
                sanitized_title = ydl.prepare_filename(info)
                final_path = os.path.splitext(sanitized_title)[0] + ".wav"
                
                return final_path

        except Exception as e:
            print(f"YouTube Download Error: {e}")
            return None

if __name__ == "__main__":
    yf = YouTubeFetcher()
    # url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Rick Roll test
    # print(yf.download_audio(url))
