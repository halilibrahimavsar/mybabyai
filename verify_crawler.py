import sys
from pathlib import Path

# Add project root to sys.path
# sys.path.insert(0, str(Path(__file__).parent))

from src.utils.web_crawler import WebCrawler
from src.utils.logger import setup_logger

def verify_crawler():
    setup_logger("test_crawler")
    crawler = WebCrawler()
    
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    print(f"Testing crawler with: {test_url}")
    
    text = crawler.fetch_wikipedia_text(test_url)
    
    if text:
        print("\n--- Crawler Success ---")
        print(f"Extracted {len(text)} characters.")
        print("First 500 characters:")
        print(text[:500])
        print("------------------------\n")
        return True
    else:
        print("\n--- Crawler Failed ---\n")
        return False

if __name__ == "__main__":
    verify_crawler()
