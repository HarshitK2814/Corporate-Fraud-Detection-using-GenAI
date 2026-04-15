
import os
import librosa
import numpy as np
import soundfile as sf

class AudioProcessor:
    """
    Analyzes audio files for micro-tremors and stress markers.
    """
    def __init__(self):
        pass

    def extract_features(self, audio_path):
        """
        Extracts acoustic features: Pitch Variance, Jitter (simulated), Shimmer (simulated),
        and Pause Rate.
        
        Note: True Jitter/Shimmer requires glottal pulse detection (Praat/Parselmouth).
        Here we use signal processing approximations using Librosa.
        """
        try:
            # Load audio (resample to 22050 Hz)
            # Load audio (resample to 22050 Hz)
            # OPTIMIZATION: Only analyze first 15 seconds (plenty for demo tokens)
            y, sr = librosa.load(audio_path, sr=22050, duration=15)
            
            # 1. Pitch (Fundamental Frequency - F0)
            # Use YIN (faster than pyin) for speed
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            # f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0 = f0[~np.isnan(f0)] # Remove NaNs

            if len(f0) == 0:
                return {"error": "No voice detected"}

            # Pitch Variance
            pitch_std = np.std(f0)
            
            # --- Speaker Normalization (Z-Score Baselining) ---
            # Literature: Forensic Phonetics / Ekman Voice Stress Detectors
            # We baseline the first 30% of the audio to establish the speaker's normal
            # resting vocal parameters, then measure max deviation (Z-score) in the rest.
            
            # 2. Jitter Approximation (Frequency Perturbation)
            jitter_frames = np.abs(np.diff(f0)) / f0[:-1]
            split_idx = max(1, len(jitter_frames) // 3)
            
            # Baseline (First 33%)
            base_jitter_mean = np.mean(jitter_frames[:split_idx])
            base_jitter_std = np.std(jitter_frames[:split_idx]) + 1e-6 # prevent div by zero
            
            # Analysis (Remaining 67%)
            if len(jitter_frames[split_idx:]) > 0:
                stress_jitter = np.mean(jitter_frames[split_idx:])
                jitter_zscore = (stress_jitter - base_jitter_mean) / base_jitter_std
            else:
                jitter_zscore = 0.0

            # 3. Shimmer Approximation (Amplitude Perturbation)
            hop_length = 512
            rmse = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
            
            shimmer_frames = np.abs(np.diff(rmse)) / (rmse[:-1] + 1e-6)
            split_idx_shimmer = max(1, len(shimmer_frames) // 3)
            
            base_shimmer_mean = np.mean(shimmer_frames[:split_idx_shimmer])
            base_shimmer_std = np.std(shimmer_frames[:split_idx_shimmer]) + 1e-6
            
            if len(shimmer_frames[split_idx_shimmer:]) > 0:
                stress_shimmer = np.mean(shimmer_frames[split_idx_shimmer:])
                shimmer_zscore = (stress_shimmer - base_shimmer_mean) / base_shimmer_std
            else:
                shimmer_zscore = 0.0

            # 4. Pause Rate (Silence / Total Time)
            non_silent_intervals = librosa.effects.split(y, top_db=20)
            non_silent_duration = sum([end - start for start, end in non_silent_intervals]) / sr
            total_duration = librosa.get_duration(y=y, sr=sr)
            pause_rate = (total_duration - non_silent_duration) / total_duration

            # Normalize Scores to 0-1 Risk (Heuristic fallback if ML fails)
            tremor_risk = 0.0
            if jitter_zscore > 2.0: tremor_risk += 0.4 # Over 2 std deviations
            if shimmer_zscore > 2.0: tremor_risk += 0.3
            if pitch_std > 50: tremor_risk += 0.3

            return {
                "duration": total_duration,
                "pitch_std": float(pitch_std),
                "jitter": float(jitter_zscore),   # Outputting Z-score now
                "shimmer": float(shimmer_zscore), # Outputting Z-score now
                "pause_rate": float(pause_rate),
                "tremor_score":  min(max(tremor_risk, 0.0), 1.0)
            }

        except Exception as e:
            print(f"Error processing audio: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    # Test
    ap = AudioProcessor()
    # print(ap.extract_features("test.wav"))
