"""
Google Street View Fetcher

Fetches street-level imagery using Google Static Street View API.
Replaces ZenSVI/Mapillary implementation.
"""

import os
import requests
import shutil
from pathlib import Path
from typing import List, Dict, Any

# We use the key from .env (YOUTUBE_API_KEY was acting as Google Key)
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

class GoogleSVFetcher:
    """
    Downloads images from Google Street View Static API.
    """
    def __init__(self, save_dir: str = "downloads/svi"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = API_KEY

    def fetch_images(
        self,
        lat: float,
        lon: float,
        company_name: str = "unknown",
        max_images: int = 4, # Standard 4 directions
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        
        company_dir = self.save_dir / company_name.replace(" ", "_").replace("/", "_") / "google"
        company_dir.mkdir(parents=True, exist_ok=True)

        # Clear cache if forced
        if force_refresh:
            print(f"    Force refresh (Google) for {company_name}...")
            for f in company_dir.glob("*"):
                f.unlink()

        downloaded_paths = []

        verified_entity = False
        found_place_name = None

        #    1. Try Google Places API (Building Photos)                    
        # This gets the actual "Place" photo (e.g. uploaded by owner/users)
        # much better than random street view for large campuses.
        print(f"    Searching Google Places for '{company_name}'...")
        
        try:
            # A. Find Place ID via Text Search
            query = f"{company_name}"
            search_url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={query}&location={lat},{lon}&radius=500&key={self.api_key}"
            
            resp = requests.get(search_url, timeout=10)
            data = resp.json()
            
            if data.get("status") == "OK" and data.get("results"):
                place = data["results"][0]
                found_place_name = place.get("name")
                print(f"    Found place: {found_place_name} ({place.get('formatted_address', '')[:30]}...)")
                
                # VERIFICATION: Stricter Name Matching
                from difflib import SequenceMatcher
                
                def similarity(a, b):
                    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

                sim_score = similarity(company_name, found_place_name)
                print(f"    Name Similarity Score: {sim_score:.2f}")

                # Threshold: 0.6 (Stricter!)
                # "Alpha" vs "Alpha Holdings" (0.43) -> FAIL
                # "Infosys" vs "Infosys Ltd" (0.8) -> PASS
                if sim_score > 0.6 or company_name.lower() in found_place_name.lower():
                    verified_entity = True
                    print(f"    Name Match Verified")
                else:
                    print(f"    Name Mismatch (Score {sim_score:.2f} < 0.6)")
                    verified_entity = False
                
                photos = place.get("photos", [])
                if photos:
                    print(f"    Found {len(photos)} photos for this place.")
                    # Download up to 3
                    for i, photo in enumerate(photos[:3]):
                        photo_ref = photo.get("photo_reference")
                        if not photo_ref: continue
                        
                        # B. Fetch Photo
                        photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=800&photo_reference={photo_ref}&key={self.api_key}"
                        
                        p_filename = f"place_{i}.jpg"
                        p_filepath = company_dir / p_filename
                        
                        # Check availability
                        if not p_filepath.exists() or force_refresh:
                            p_resp = requests.get(photo_url, timeout=15)
                            if p_resp.status_code == 200:
                                with open(p_filepath, "wb") as f:
                                    f.write(p_resp.content)
                                downloaded_paths.append(str(p_filepath.resolve()))
                            else:
                                print(f"    Failed to download photo {i}: {p_resp.status_code}")
                        else:
                             downloaded_paths.append(str(p_filepath.resolve()))
                else:
                    print("    Place found, but it has no photos.")
            else:
                print(f"    Place search failed or no results: {data.get('status')}")

        except Exception as e:
            print(f"    Google Places lookup failed: {e}")

        #    2. Fallback: Web Image Search (If Place photos invalid or few)   
        # If we didn't verify the entity, OR we have few photos, try Web Search
        if not verified_entity or len(downloaded_paths) < 2:
             try:
                from modules.geospatial.web_image_fetcher import WebImageFetcher
                web_fetcher = WebImageFetcher(save_dir=self.save_dir)
                web_query = f"{company_name} building exterior"
                print(f"    Fetching web images for '{web_query}'...")
                
                web_images = web_fetcher.fetch_images(web_query, company_name, max_images=2)
                if web_images:
                    # If we found web images, valid? Maybe.
                    print(f"    Added {len(web_images)} web images.")
                    web_paths = [Path(p) for p in web_images]
                    
                    # Convert to string and append
                    for p in web_paths:
                        downloaded_paths.insert(0, str(p)) # Prioritize? Or append?
             except Exception as e:
                print(f"    Web search failed: {e}")


        #    3. Fallback/Supplement: Google Street View (Coordinates)       
        # If we have < 4 images, fill up with Street View
        if len(downloaded_paths) < max_images:
            needed = max_images - len(downloaded_paths)
            print(f"    Supplementing with {needed} Street View images...")
            
            headings = [0, 90, 180, 270] # N, E, S, W
            # If we need fewer, just take first N
            current_headings = headings[:needed] if needed < 4 else headings

            for h in current_headings:
                filename = f"gsv_{lat}_{lon}_{h}.jpg"
                filepath = company_dir / filename
                
                url = f"https://maps.googleapis.com/maps/api/streetview?size=640x640&location={lat},{lon}&fov=90&heading={h}&pitch=10&key={self.api_key}"
                
                try:
                    if not filepath.exists() or force_refresh:
                        resp = requests.get(url, timeout=10)
                        if resp.status_code == 200:
                            with open(filepath, "wb") as f:
                                f.write(resp.content)
                            downloaded_paths.append(str(filepath.resolve()))
                    else:
                        downloaded_paths.append(str(filepath.resolve()))
                except Exception as e:
                    print(f"    SV Download failed: {e}")

        return {
            "source": "google_hybrid",
            "num_images": len(downloaded_paths),
            "image_paths": downloaded_paths,
            "metadata": {
                "lat": lat, "lon": lon, 
                "verified_entity": verified_entity,
                "found_place_name": found_place_name,
                "match_score": sim_score if 'sim_score' in locals() else 0.0
            }
        }

# Alias for app.py compatibility
SVIFetcher = GoogleSVFetcher

if __name__ == "__main__":
    import sys
    f = GoogleSVFetcher()
    # Test coords
    f.fetch_images(12.851593, 77.664197, "Test_Google")
