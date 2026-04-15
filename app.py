from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Import our modules
from modules.geospatial.mapillary_fetcher import SVIFetcher
from modules.geospatial.building_classifier import BuildingClassifier
# from modules.voice.audio_fetcher import AudioFetcher 
# from modules.voice.transcriber import Transcriber
# from modules.voice.forensics import AudioForensics
# from modules.fusion.score_generator import ScoreGenerator
# (Uncomment as we integrate them)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env
load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Global instances (lazy load or init here)
# For demo, we might want to init them lazy to speed up startup during dev
svi_fetcher = None
building_classifier = None

def get_svi_fetcher():
    global svi_fetcher
    if svi_fetcher is None:
        svi_fetcher = SVIFetcher()
    return svi_fetcher

def get_building_classifier():
    global building_classifier
    if building_classifier is None:
        building_classifier = BuildingClassifier()
    return building_classifier

#    Routes                                                                   

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/verify_geo', methods=['POST'])
def verify_geo():
    """Step 1: Verify Geospatial Location & Fetch Images"""
    data = request.json
    company = data.get('company', 'Unknown')
    lat = float(data.get('lat'))
    lon = float(data.get('lon'))
    
    logger.info(f"Verifying geo for {company} at {lat}, {lon}")
    
    fetcher = get_svi_fetcher()
    # Always force refresh when user clicks 'Verify' to ensure new coords are used
    # Reduce max_images to 5 for speed (demo mode)
    result = fetcher.fetch_images(lat, lon, company_name=company, max_images=5, force_refresh=True)
    
    # Return result with image paths accessible via static or special route
    # We need to serve downloaded images. 
    # fetcher saves to 'downloads/svi/...'
    # We can create a route to serve them.
    
    # Convert absolute paths to relative URLs for frontend
    frontend_images = []
    if result.get('image_paths'):
        for p in result['image_paths']:
            # Assuming downloads is in project root
            # We need to expose 'downloads' dir via Flask
            rel_path = os.path.relpath(p, os.getcwd())
            frontend_images.append(f"/files/{rel_path.replace(os.path.sep, '/')}")
            
    result['image_urls'] = frontend_images
    del result['image_paths'] # remove absolute paths
    
    return jsonify(result)

@app.route('/api/classify_building', methods=['POST'])
def classify_building():
    """Step 2: Classify Fetched Images"""
    data = request.json
    image_urls = data.get('image_urls', [])
    # SECURITY FIX: Default to False if missing. If frontend doesn't send it, we assume UNVERIFIED.
    verified = data.get('verified_entity', False) 
    
    logger.info(f"Classifying {len(image_urls)} images. Entity Verified: {verified}")

    if not image_urls:
         return jsonify({"error": "No images to classify"}), 400

    # Convert URLs back to local paths
    image_paths = []
    for url in image_urls:
        # /files/downloads/svi/... -> downloads/svi/...
        local_path = url.replace('/files/', '').replace('/', os.path.sep)
        image_paths.append(local_path)

    classifier = get_building_classifier()
    verdict = classifier.classify_batch(image_paths, verified_entity=verified)
    
    return jsonify(verdict)

# Route to serve dynamic files (downloads)
@app.route('/files/<path:filename>')
def serve_files(filename):
    return send_from_directory('.', filename)

#    Step 4: Voice Analysis Endpoint                                          

from modules.voice.audio_processor import AudioProcessor
from modules.voice.text_analyzer import TextAnalyzer
from modules.voice.youtube_fetcher import YouTubeFetcher
# For transcription, we'll use Groq (Whisper) or a simple mock if needed.
# Since we have Groq, we can use it for transcription if the model supports it (distil-whisper)
# OR we can just use the 'transcript_text' input for now as the primary method for text analysis.
# For audio, we extract acoustic features.

@app.route('/api/analyze_voice', methods=['POST'])
def analyze_voice():
    """Step 4: Analyze Voice (Audio or Text)"""
    try:
        data = request.form
        input_type = data.get('input_type', 'upload') # upload, youtube, text
        
        audio_path = None
        transcript_text = ""
        transcription_source = "Manual Input"

        # 1. Handle Input Source
        if input_type == 'youtube':
            url = data.get('youtube_url')
            fetcher = YouTubeFetcher()
            audio_path = fetcher.download_audio(url)
            if not audio_path:
                return jsonify({"error": "Failed to download YouTube audio"}), 400
            transcription_source = "YouTube Audio"

        elif input_type == 'upload':
            if 'file' not in request.files:
                return jsonify({"error": "No file uploaded"}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Save uploaded file
            save_dir = Path("downloads/audio")
            save_dir.mkdir(parents=True, exist_ok=True)
            audio_path = str(save_dir / file.filename)
            file.save(audio_path)
            transcription_source = "Uploaded File"

        elif input_type == 'text':
            transcript_text = data.get('transcript_text', '')
            if not transcript_text:
                return jsonify({"error": "No text provided"}), 400

        # 2. Analyze Audio (Micro-Tremors)
        audio_results = {}
        if audio_path:
            logger.info(f"Analyzing Audio: {audio_path}")
            ap = AudioProcessor()
            audio_results = ap.extract_features(audio_path)
            
            # TODO: Transcribe audio to text for semantic analysis
            # For now, if no text provided, we might skip semantic analysis 
            # OR use a placeholder if we don't have a transcriber set up yet.
            # Let's assume the user provides text OR we focus on acoustic only for audio.
            if not transcript_text:
                transcript_text = "(Audio transcription not yet implemented. Acoustic analysis only.)"

        # 3. Analyze Text (Semantic Drift)
        text_results = {}
        if transcript_text and len(transcript_text) > 20:
             logger.info("Analyzing Text Semantics...")
             ta = TextAnalyzer()
             text_results = ta.analyze_semantics(transcript_text)
        
        # 4. Fusion / Response
        return jsonify({
            "input_type": input_type,
            "audio_analysis": audio_results,
            "text_analysis": text_results,
            "transcription": transcript_text
        })

    except Exception as e:
        logger.error(f"Voice Analysis Error: {e}")
        return jsonify({"error": str(e)}), 500

#    Step 5: Fusion Score Endpoint                                           

from modules.fusion.score_generator import FusionScorer

@app.route('/api/generate_score', methods=['POST'])
def generate_score():
    """Step 5: Generate Final CRDI Score"""
    try:
        data = request.json
        company = data.get('company', 'Unknown')
        
        # Reconstruct inputs from frontend state
        # In a real app, these would be saved in DB. Here we trust frontend to pass them back
        # OR we just rely on what we have. 
        # Actually, let's accept the risk scores directly from frontend for simplicity in this demo
        # or implies we have state.
        
        # Better: Frontend sends the partial results it has.
        geospatial_result = data.get('geospatial_result') # { avg_shell_risk: ... }
        behavioral_result = data.get('behavioral_result') # { combined_behavioral_score: ... }
        
        # If frontend sends raw data, we might need to structure it.
        # Let's assume frontend sends:
        # geo_risk (0-1), voice_risk (0-1), etc.
        
        # To use FusionScorer properly, we need to mimic its expected inputs.
        # It expects Dicts.
        
        scorer = FusionScorer()
        report = scorer.generate_score(
            company_name=company,
            geospatial_result=geospatial_result,
            behavioral_result=behavioral_result
        )
        
        return jsonify(report)

    except Exception as e:
        logger.error(f"Fusion Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure downloads dir exists
    Path("downloads").mkdir(exist_ok=True)
    app.run(debug=True, port=5000)
