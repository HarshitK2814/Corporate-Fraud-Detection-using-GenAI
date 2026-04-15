"""
Voice Forensics   Micro-Tremor Detection + Semantic Drift Analysis

Two independent signals:
1. **Audio Micro-Tremor**: Extracts jitter, shimmer, pitch variability
   from the audio signal as physiological stress indicators.
2. **Semantic Drift**: Uses Groq (LLaMA) to detect hedging, evasion,
   and contradiction patterns in the transcript.
"""

import json
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Any, List, Optional
from groq import Groq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import GROQ_API_KEY, GROQ_MODEL


#                                                                            
#  Part 1   Audio Micro-Tremor Detection
#                                                                            

class MicroTremorDetector:
    """
    Extracts vocal biomarkers from audio that correlate with stress/deception.

    Metrics:
        - jitter:  cycle-to-cycle pitch period variation (involuntary)
        - shimmer: cycle-to-cycle amplitude variation
        - pitch_std: overall pitch instability
        - hnr: harmonics-to-noise ratio (voice quality)
        - speech_rate_var: variability in speech pace
    """

    def analyze(self, audio_path: str | Path, sr: int = 22050) -> Dict[str, Any]:
        """
        Analyse an audio file and return stress indicators.
        """
        audio_path = Path(audio_path)
        # Limit analysis to first 5 minutes (300s) to avoid excessive processing time
        y, sr = librosa.load(str(audio_path), sr=sr, duration=300)

        #    Pitch (F0) extraction                                        
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        f0_voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])

        #    Jitter (pitch period perturbation)                           
        jitter = 0.0
        if len(f0_voiced) > 1:
            periods = 1.0 / f0_voiced
            jitter = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods))

        #    Shimmer (amplitude perturbation)                             
        shimmer = 0.0
        rms = librosa.feature.rms(y=y)[0]
        if len(rms) > 1:
            shimmer = float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-8))

        #    Pitch statistics                                             
        pitch_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        pitch_std = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        pitch_cv = pitch_std / (pitch_mean + 1e-8)

        #    Harmonics-to-Noise Ratio                                     
        # Approximate via spectral flatness (flat = noisy = low HNR)
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        hnr_approx = float(-10 * np.log10(np.mean(flatness) + 1e-10))

        #    Speech rate variability                                      
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        speech_rate_var = 0.0
        if len(onsets) > 2:
            intervals = np.diff(onsets)
            speech_rate_var = float(np.std(intervals) / (np.mean(intervals) + 1e-8))

        #    Composite stress score                                       
        # Higher = more stress indicators
        stress_score = self._compute_stress_score(
            jitter, shimmer, pitch_cv, hnr_approx, speech_rate_var
        )

        return {
            "jitter": round(jitter, 6),
            "shimmer": round(shimmer, 6),
            "pitch_mean_hz": round(pitch_mean, 2),
            "pitch_std_hz": round(pitch_std, 2),
            "pitch_cv": round(pitch_cv, 4),
            "hnr_approx_db": round(hnr_approx, 2),
            "speech_rate_variability": round(speech_rate_var, 4),
            "stress_score": round(stress_score, 4),
            "duration_sec": round(len(y) / sr, 2),
        }

    @staticmethod
    def _compute_stress_score(jitter, shimmer, pitch_cv, hnr, speech_var) -> float:
        """
        Weighted composite of stress indicators. Range 0 1.
        Thresholds calibrated for BROADCAST / YouTube audio
        (compression artifacts, background noise, multi-speaker panels).
        """
        scores = []

        # Jitter: broadcast normal ~3-5%, stressed > 8%
        scores.append(min(jitter / 0.08, 1.0) * 0.25)

        # Shimmer: broadcast normal ~8-15%, stressed > 25%
        scores.append(min(shimmer / 0.25, 1.0) * 0.20)

        # Pitch CV: multi-speaker calls normal ~0.3, stressed > 0.6
        scores.append(min(pitch_cv / 0.60, 1.0) * 0.20)

        # HNR: broadcast normal ~10-15dB (vs 20+ clinical)
        hnr_norm = max(0, 1 - (hnr / 15.0))
        scores.append(hnr_norm * 0.15)

        # Speech rate: multi-speaker panels have high variability naturally
        scores.append(min(speech_var / 1.0, 1.0) * 0.20)

        return sum(scores)


#                                                                            
#  Part 2   Semantic Drift Analysis (Groq / LLaMA)
#                                                                            

class SemanticDriftAnalyzer:
    """
    Analyses earnings call transcripts for linguistic deception markers:
    - Hedging & vague language
    - Contradictions across segments
    - Evasion of direct questions
    - Excessive optimism / deflection
    """

    SYSTEM_PROMPT = """You are a forensic linguistics expert analysing CEO earnings call transcripts.

Your job is to detect GENUINE deception, evasion, and manipulation in corporate communications.

CALIBRATION RULES (critical   follow these strictly):
- Earnings calls NORMALLY contain positive framing. Score optimism_bias ONLY for
  unrealistic claims unsupported by data, or downplaying of clearly negative results.
- Phrases like "we believe", "going forward", "strong performance", "well positioned"
  are STANDARD corporate language, NOT hedging. Only flag hedging when it is used
  to AVOID providing specific numbers or commitments.
- Score 0-25 = normal corporate communication. Reserve 50+ for genuine red flags.
- Contradiction requires SPECIFIC conflicting statements within the same transcript,
  not just mixed sentiment or balanced pros/cons.
- Evasion requires a QUESTION being asked and NOT answered directly.
- If the transcript is a prepared statement (not Q&A), evasion_score should be low.

Analyse the provided transcript chunk and return a JSON object with these fields:
{
    "hedging_score": 0-100,
    "contradiction_score": 0-100,
    "evasion_score": 0-100,
    "optimism_bias_score": 0-100,
    "key_red_flags": ["..."],
    "summary": "..."
}

Be rigorous and specific. Cite exact phrases as evidence. Return ONLY valid JSON, no markdown."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or GROQ_API_KEY
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY in .env")
        self.model = model or GROQ_MODEL
        self.client = Groq(api_key=self.api_key)

    def analyze_chunk(self, text: str) -> Dict[str, Any]:
        """Analyse a single transcript chunk."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analyse this transcript:\n\n{text[:4000]}"},
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"error": "Failed to parse LLM response", "raw": raw[:500]}
        except Exception as e:
            return {"error": str(e)}

    def analyze_full_transcript(
        self, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyse all transcript chunks and detect drift across time.

        Args:
            chunks: list of {"start": float, "end": float, "text": str}

        Returns aggregated scores + per-chunk results.
        """
        chunk_results = []
        for i, chunk in enumerate(chunks):
            print(f"    Analysing chunk {i+1}/{len(chunks)} ({chunk['start']:.0f}s {chunk['end']:.0f}s)")
            analysis = self.analyze_chunk(chunk["text"])
            analysis["time_start"] = chunk["start"]
            analysis["time_end"] = chunk["end"]
            chunk_results.append(analysis)

        # Aggregate scores
        valid = [c for c in chunk_results if "hedging_score" in c]
        if not valid:
            return {
                "chunk_analyses": chunk_results,
                "overall": {"error": "No valid analyses"},
            }

        avg = lambda key: sum(c.get(key, 0) for c in valid) / len(valid)

        # Detect drift: are scores increasing over time? (fatigue / pressure)
        hedging_trend = self._compute_trend([c.get("hedging_score", 0) for c in valid])
        evasion_trend = self._compute_trend([c.get("evasion_score", 0) for c in valid])

        all_flags = []
        for c in valid:
            all_flags.extend(c.get("key_red_flags", []))

        overall_deception_score = (
            avg("hedging_score") * 0.25 +
            avg("contradiction_score") * 0.30 +
            avg("evasion_score") * 0.30 +
            avg("optimism_bias_score") * 0.15
        ) / 100  # normalize to 0-1

        return {
            "chunk_analyses": chunk_results,
            "overall": {
                "avg_hedging": round(avg("hedging_score"), 1),
                "avg_contradiction": round(avg("contradiction_score"), 1),
                "avg_evasion": round(avg("evasion_score"), 1),
                "avg_optimism_bias": round(avg("optimism_bias_score"), 1),
                "hedging_trend": hedging_trend,
                "evasion_trend": evasion_trend,
                "deception_score": round(overall_deception_score, 4),
                "key_red_flags": all_flags[:10],  # top 10
                "num_chunks_analyzed": len(valid),
            },
        }

    @staticmethod
    def _compute_trend(values: List[float]) -> str:
        """Simple trend: compare first half vs second half average."""
        if len(values) < 2:
            return "STABLE"
        mid = len(values) // 2
        first = sum(values[:mid]) / max(mid, 1)
        second = sum(values[mid:]) / max(len(values) - mid, 1)
        diff = second - first
        if diff > 10:
            return "INCREASING"   # getting worse
        elif diff < -10:
            return "DECREASING"
        return "STABLE"


#                                                                            
#  Combined Forensics Interface
#                                                                            

class VoiceForensics:
    """
    Combines micro-tremor (audio) + semantic drift (text) analysis.
    """

    def __init__(self):
        self.tremor = MicroTremorDetector()
        self.semantic = SemanticDriftAnalyzer()

    def full_analysis(
        self,
        audio_path: str | Path,
        transcript_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run both audio and text analyses and merge results.
        """
        print("\n   Audio Micro-Tremor Analysis   ")
        audio_result = self.tremor.analyze(audio_path)

        print("\n   Semantic Drift Analysis   ")
        semantic_result = self.semantic.analyze_full_transcript(transcript_chunks)

        # Combine into a single behavioural score
        audio_stress = audio_result.get("stress_score", 0)
        semantic_deception = semantic_result.get("overall", {}).get("deception_score", 0)

        combined_score = audio_stress * 0.40 + semantic_deception * 0.60

        if combined_score < 0.25:
            verdict = "LOW_RISK"
        elif combined_score < 0.50:
            verdict = "MODERATE_RISK"
        elif combined_score < 0.75:
            verdict = "HIGH_RISK"
        else:
            verdict = "CRITICAL_RISK"

        return {
            "audio_analysis": audio_result,
            "semantic_analysis": semantic_result.get("overall", {}),
            "chunk_details": semantic_result.get("chunk_analyses", []),
            "combined_behavioral_score": round(combined_score, 4),
            "verdict": verdict,
        }


#    Quick test                                                               
if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) < 2:
        print("Usage: python forensics.py <audio_file>")
        print("  (also needs corresponding transcript chunks)")
        _sys.exit(1)

    detector = MicroTremorDetector()
    result = detector.analyze(_sys.argv[1])
    print("\n   Micro-Tremor Results   ")
    for k, v in result.items():
        print(f"  {k:30s}: {v}")
