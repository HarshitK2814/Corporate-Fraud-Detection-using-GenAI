"""
Whisper Transcriber

Transcribes audio files using OpenAI Whisper (runs locally).
Produces timestamped segments for downstream forensic analysis.
"""

import whisper
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import WHISPER_MODEL_SIZE


class Transcriber:
    """
    Local Whisper transcription.

    Usage:
        t = Transcriber()
        result = t.transcribe("path/to/audio.wav")
        print(result["text"])
    """

    def __init__(self, model_size: str = None):
        size = model_size or WHISPER_MODEL_SIZE
        print(f"  Loading Whisper model ({size}) ...")
        self.model = whisper.load_model(size)

    def transcribe(
        self,
        audio_path: str | Path,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.

        Returns:
            {
                "text": str,           # full transcript
                "segments": [          # timestamped chunks
                    {"start": float, "end": float, "text": str},
                    ...
                ],
                "language": str,
                "duration_sec": float,
            }
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"    Transcribing {audio_path.name} ...")
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            verbose=False,
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip(),
            })

        duration = segments[-1]["end"] if segments else 0.0

        return {
            "text": result["text"].strip(),
            "segments": segments,
            "language": result.get("language", language),
            "duration_sec": round(duration, 2),
        }

    def transcribe_with_chunks(
        self,
        audio_path: str | Path,
        chunk_duration_sec: float = 120.0,
    ) -> List[Dict[str, Any]]:
        """
        Transcribe and group segments into larger time-chunks
        for semantic drift analysis (e.g., 2-minute windows).
        """
        result = self.transcribe(audio_path)
        segments = result["segments"]
        if not segments:
            return []

        chunks: List[Dict[str, Any]] = []
        current_chunk_text = []
        chunk_start = segments[0]["start"]

        for seg in segments:
            current_chunk_text.append(seg["text"])
            if seg["end"] - chunk_start >= chunk_duration_sec:
                chunks.append({
                    "start": round(chunk_start, 2),
                    "end": round(seg["end"], 2),
                    "text": " ".join(current_chunk_text),
                })
                current_chunk_text = []
                chunk_start = seg["end"]

        # Last chunk
        if current_chunk_text:
            chunks.append({
                "start": round(chunk_start, 2),
                "end": round(segments[-1]["end"], 2),
                "text": " ".join(current_chunk_text),
            })

        return chunks


#    Quick test                                                               
if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) < 2:
        print("Usage: python transcriber.py <audio_file>")
        _sys.exit(1)

    t = Transcriber()
    r = t.transcribe(_sys.argv[1])
    print(f"\nDuration: {r['duration_sec']:.0f}s | Segments: {len(r['segments'])}")
    print(f"\n{r['text'][:500]}...")
