"""
Veritas AI   Main Orchestrator

End-to-end pipeline:
  1. Fetch street-level images via Mapillary    classify buildings
  2. Fetch earnings call audio via YouTube      transcribe   analyse
  3. Fuse both signals into a CRDI Truth Score
"""

import argparse
import json
from pathlib import Path

from modules.geospatial.mapillary_fetcher import SVIFetcher
from modules.geospatial.building_classifier import BuildingClassifier
# ...

def run_geospatial(lat: float, lon: float, company_name: str) -> dict:
    """Run the geospatial shell-company detection pipeline."""
    print(f"\n{'='*60}")
    print(f"    Geospatial Verification: {company_name}")
    print(f"  Coordinates: ({lat}, {lon})")
    print(f"{'='*60}")

    fetcher = SVIFetcher()
    # ZenSVI fetcher takes company_name for directory organization
    fetch_result = fetcher.fetch_images(lat, lon, company_name=company_name)

    if not fetch_result["image_paths"]:
        print("    No images found   skipping classification")
        return None

    classifier = BuildingClassifier()
    result = classifier.classify_batch(fetch_result["image_paths"])

    print(f"\n  Verdict: {result['verdict']} (avg risk: {result['avg_shell_risk']:.2%})")
    return result

    print(f"\n  Verdict: {result['verdict']} (avg risk: {result['avg_shell_risk']:.2%})")
    return result


def run_voice_analysis(company_name: str, quarter: str = "") -> dict:
    """Run the CEO voice forensics pipeline."""
    print(f"\n{'='*60}")
    print(f"    Voice Forensics: {company_name} {quarter}")
    print(f"{'='*60}")

    # Step 1: Fetch audio
    audio_fetcher = AudioFetcher()
    audio_path = audio_fetcher.fetch_earnings_call(company_name, quarter)

    if not audio_path:
        print("    No audio available   skipping voice analysis")
        return None

    # Step 2: Transcribe
    transcriber = Transcriber()
    chunks = transcriber.transcribe_with_chunks(audio_path, chunk_duration_sec=120)

    if not chunks:
        print("    Transcription produced no chunks")
        return None

    print(f"    Transcribed into {len(chunks)} chunks")

    # Step 3: Forensic analysis
    forensics = VoiceForensics()
    result = forensics.full_analysis(audio_path, chunks)

    print(f"\n  Verdict: {result['verdict']} (combined score: {result['combined_behavioral_score']:.2%})")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Veritas AI   Corporate Reality Distortion Intelligence"
    )
    parser.add_argument("--company", required=True, help="Company name (e.g., 'Infosys')")
    parser.add_argument("--lat", type=float, help="Latitude of registered address")
    parser.add_argument("--lon", type=float, help="Longitude of registered address")
    parser.add_argument("--quarter", default="", help="Earnings call quarter (e.g., 'Q3 2024')")
    parser.add_argument("--skip-geo", action="store_true", help="Skip geospatial analysis")
    parser.add_argument("--skip-voice", action="store_true", help="Skip voice analysis")
    parser.add_argument("--output", help="Save JSON report to file")

    args = parser.parse_args()

    geo_result = None
    voice_result = None

    #    Geospatial                                                       
    if not args.skip_geo:
        if args.lat is not None and args.lon is not None:
            geo_result = run_geospatial(args.lat, args.lon, args.company)
        else:
            print("    Lat/Lon not provided   skipping geospatial analysis")

    #    Voice                                                            
    if not args.skip_voice:
        voice_result = run_voice_analysis(args.company, args.quarter)

    #    Fusion                                                           
    scorer = FusionScorer()
    crdi = scorer.generate_score(args.company, geo_result, voice_result)
    report = FusionScorer.format_report(crdi)
    print(report)

    # Save report
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = DATA_DIR / f"crdi_{args.company.replace(' ', '_')}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(crdi, f, indent=2, default=str)
    print(f"\n    Report saved to: {out_path}")


if __name__ == "__main__":
    main()
