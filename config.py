"""
Veritas AI - Configuration
Loads API keys from .env and provides project-wide settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

# Ensure ffmpeg is available
# try:
#     import static_ffmpeg
#     static_ffmpeg.add_paths()
# except ImportError:
#     print("Warning: static-ffmpeg not found. Audio processing may fail if ffmpeg is missing.")


#    API Keys                                                                 
MAPILLARY_ACCESS_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN", "")
MAPILLARY_TOKEN = MAPILLARY_ACCESS_TOKEN  # Alias for compatibility
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

#    Mapillary Settings                                                       
MAPILLARY_API_BASE = "https://graph.mapillary.com"
MAPILLARY_SEARCH_RADIUS_M = 100          # meters
MAPILLARY_MAX_IMAGES = 5                 # images per query

#    Groq Settings                                                            
GROQ_MODEL = "llama-3.3-70b-versatile"   # fast + capable
GROQ_API_BASE = "https://api.groq.com/openai/v1"

#    Whisper Settings                                                         
WHISPER_MODEL_SIZE = "base"              # tiny | base | small | medium | large

#    Paths                                                                    
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
DOWNLOADS_DIR.mkdir(exist_ok=True)
