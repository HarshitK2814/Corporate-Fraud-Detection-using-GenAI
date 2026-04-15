"""
Building Classifier   Places365 (PyTorch Version)

Classifies street-view images into scene categories using a pre-trained
ResNet18-Places365 model. Maps categories to shell-company risk scores.

Replaces zensvi.cv due to onnxruntime DLL issues on Windows.
"""

import os
import csv
import torch
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Union
from PIL import Image
from torchvision import transforms, models

from config import DATA_DIR, DOWNLOADS_DIR

#    Scene Risk Mapping                                                        
# Maps Places365 categories to risk scores (0.0 = Corporate, 1.0 = High Risk)
SCENE_RISK_MAPPING = {
    # Low Risk (Corporate/Office/Modern)
    "office_building": 0.05,
    "skyscraper": 0.05,
    "campus": 0.10,
    "conference_center": 0.10,
    "hospital": 0.10,
    "museum": 0.10,
    "museum/outdoor": 0.10,  # Exact match fix
    "museum/indoor": 0.10,
    "embassy": 0.10,
    "arena": 0.15,
    "sport_arena": 0.15,     # Places365 label
    "baseball_field": 0.15,
    "football_field": 0.15,
    "stadium": 0.15,
    "soccer_stadium": 0.15,
    "pavilion": 0.15,
    "hotel": 0.20,
    "hotel/outdoor": 0.20,
    "plaza": 0.20,
    "courtyard": 0.20,
    "fountain": 0.20,
    "lobby": 0.10,
    
    # Medium Risk (Industrial/Commercial/Public)
    "industrial_area": 0.30,
    "warehouse": 0.30,
    "factory": 0.30,
    "shopping_mall": 0.40,
    "supermarket": 0.40,
    "market": 0.45,
    "market/outdoor": 0.45,
    "promenade": 0.40,
    "street": 0.50,          # Neutral
    "crosswalk": 0.50,
    "highway": 0.50,
    "downtown": 0.15,        # Usually corporate
    
    # High Risk (Residential/Vacant/Chaotic)
    "residential_neighborhood": 0.60,
    "apartment_building/outdoor": 0.65,
    "apartment_building_outdoor": 0.65, # Just in case
    "medina": 0.80,          
    "bazaar": 0.70,
    "bazaar/outdoor": 0.70,
    "house": 0.70,
    "parking_lot": 0.85,
    "vacant_lot": 0.90,
    "field": 0.90,
    "construction_site": 0.80,
    "alley": 0.85,
    "slum": 0.95,
}

DEFAULT_RISK = 0.50

class BuildingClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = DOWNLOADS_DIR / "models" / "places365"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.weights_path = self.model_dir / "resnet18_places365.pth.tar"
        self.labels_path = self.model_dir / "categories_places365.txt"
        
        self.model = None
        self.classes = []
        self._setup_model()

    def _setup_model(self):
        """Load model and weights, downloading if necessary."""
        # 1. Download files
        if not self.weights_path.exists():
            print("    Downloading Places365 weights (ResNet18)...")
            url = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
            urllib.request.urlretrieve(url, self.weights_path)
            
        if not self.labels_path.exists():
            print("    Downloading Places365 labels...")
            url = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
            urllib.request.urlretrieve(url, self.labels_path)

        # 2. Load Labels
        self.classes = []
        with open(self.labels_path) as f:
            for line in f:
                # Format: /a/airfield 0
                parts = line.strip().split(' ')
                # Some lines might have different format? usually /x/y 0
                # We strip the leading /x/ part if we want?
                # Actually, Places365 classes are like /a/airfield, /a/apartment_building/outdoor
                # My logic: parts[0][3:] cuts off /a/ 
                # So /a/apartment_building/outdoor -> apartment_building/outdoor
                category = parts[0][3:] 
                self.classes.append(category)

        # 3. Load Model
        # ResNet18 with 365 output classes
        self.model = models.resnet18(num_classes=365)
        
        # Load state dict
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        state_dict = {str(k).replace('module.', ''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print(f"    Loaded Places365-ResNet18 on {self.device}")

    def classify_batch(self, image_paths: List[Union[str, Path]], verified_entity: bool = True) -> Dict[str, Any]:
        """Classify a batch of images and return risk summary."""
        if not image_paths:
            return {"verdict": "UNKNOWN", "avg_shell_risk": DEFAULT_RISK, "individual": []}

        results = []
        # We don't track simple sum anymore, we analyze the distribution
        
        print(f"    Classifying {len(image_paths)} images with Places365...")
        
        with torch.no_grad():
            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    input_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    outputs = self.model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    score, idx = torch.max(probs, 1)
                    
                    category = self.classes[idx.item()]
                    confidence = score.item()
                    
                    # Improved Lookup: Try exact, then partial
                    risk = SCENE_RISK_MAPPING.get(category)
                    if risk is None:
                        # Fallback: check if known keywords are in the category string
                        # e.g. "office" in "office_cubicles"
                        found_partial = False
                        for key, val in SCENE_RISK_MAPPING.items():
                            if key in category and len(key) > 4: # avoid matching "a" or short
                                risk = val
                                found_partial = True
                                break
                        if not found_partial:
                            risk = DEFAULT_RISK
                    
                    results.append({
                        "image": str(img_path),
                        "top_label": category,
                        "confidence": confidence,
                        "shell_risk_score": risk
                    })
                    print(f"    - {Path(img_path).name}: {category} ({confidence:.2f}) -> Risk: {risk}")
                    
                except Exception as e:
                    print(f"    Error classifying {img_path}: {e}")
                    results.append({
                        "image": str(img_path),
                        "top_label": "error",
                        "confidence": 0.0,
                        "shell_risk_score": DEFAULT_RISK
                    })

        if not results:
             return {"verdict": "UNKNOWN", "avg_shell_risk": DEFAULT_RISK, "individual": []}

        #    Advanced Verdict Logic                                       
        # 0. Check Entity Verification (from Google Places Name Match)
        if not verified_entity:
            print("    Entity NOT verified at this location. High Risk.")
            return {
                "verdict": "HIGH RISK (Entity Not Found)",
                "avg_shell_risk": 0.95,
                "individual": results
            }

        # 1. Identify valid corporate signals
        # If we have at least one image with (risk < 0.25) and decent confidence,
        # we consider it a STRONG candidate for "Corporate".
        
        min_risk = min([r['shell_risk_score'] for r in results])
        avg_risk_raw = sum([r['shell_risk_score'] for r in results]) / len(results)
        
        final_score = avg_risk_raw
        
        if min_risk <= 0.25:
            # We found something that looks like an office/campus/museum.
            # Filter out "High Risk" outliers (slums, vacant lots) that might be from bad angles.
            # We only keep images with Risk < 0.60 for the average calculation.
            filtered_risks = [r['shell_risk_score'] for r in results if r['shell_risk_score'] < 0.60]
            
            if filtered_risks:
                # Recalculate average without the "slum" noise
                adj_avg = sum(filtered_risks) / len(filtered_risks)
                print(f"    Corporate signal detected! Ignoring high-risk noise. Adj Avg: {adj_avg:.2f}")
                
                # Further bias towards the best image
                final_score = (min_risk * 0.7) + (adj_avg * 0.3)
            else:
                 # Should not happen if min_risk < 0.25
                 final_score = min_risk
        
        # Verdict mapping
        if final_score < 0.40:
            verdict = "LOW RISK (Corporate Infrastructure)"
        elif final_score < 0.70:
            verdict = "MEDIUM RISK (Mixed/Unknown)"
        else:
            verdict = "HIGH RISK (Residential/Vacant)"

        return {
            "verdict": verdict,
            "avg_shell_risk": final_score,
            "individual": results
        }

