from duckduckgo_search import DDGS
from pathlib import Path
import requests
import time

class WebImageFetcher:
    def __init__(self, download_dir="downloads/svi"):
        self.download_dir = Path(download_dir)
        self.ddgs = DDGS()

    def fetch_images(self, query: str, company_name: str, max_images: int = 3):
        """
        Fetch images from DuckDuckGo for a given query (e.g., 'Infosys Electronic City building').
        """
        company_dir = self.download_dir / company_name.replace(" ", "_").replace("/", "_") / "web"
        company_dir.mkdir(parents=True, exist_ok=True)

        print(f"    Searching web images for '{query}'...")
        
        try:
            results = self.ddgs.images(
                keywords=query,
                region="wt-wt",
                safesearch="off",
                max_results=10,
            )
            
            downloaded_paths = []
            count = 0
            
            for r in results:
                if count >= max_images:
                    break
                
                url = r.get('image')
                if not url:
                    continue
                
                try:
                    # Download image
                    print(f"      Downloading: {url[:50]}...")
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        # Guess extension or default to jpg
                        ext = "jpg"
                        if "png" in url.lower(): ext = "png"
                        if "jpeg" in url.lower(): ext = "jpg"
                        
                        filename = f"web_{int(time.time())}_{count}.{ext}"
                        filepath = company_dir / filename
                        
                        with open(filepath, "wb") as f:
                            f.write(resp.content)
                        
                        downloaded_paths.append(str(filepath.resolve()))
                        count += 1
                except Exception as e:
                    print(f"      Failed to download {url}: {e}")
                    continue
            
            return downloaded_paths

        except Exception as e:
            print(f"    Web image search failed: {e}")
            return []

if __name__ == "__main__":
    # Test
    fetcher = WebImageFetcher()
    paths = fetcher.fetch_images("Infosys Electronic City Bangalore building", "Infosys_Test")
    print("Downloaded:", paths)
