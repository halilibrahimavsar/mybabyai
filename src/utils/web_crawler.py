import requests
from bs4 import BeautifulSoup
import re
from typing import List, Optional
from src.utils.logger import get_logger

class WebCrawler:
    """Utility for extracting clean training text from Wikipedia and other websites."""
    
    def __init__(self):
        self.logger = get_logger("web_crawler")
        self.headers = {
            "User-Agent": "MyBabyAI-Crawler/1.0 (Educational Tool; +https://github.com/user/mybabyai)"
        }

    def fetch_wikipedia_text(self, url: str) -> Optional[str]:
        """Extracts and cleans main content text from a Wikipedia page."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove unwanted elements
            for element in soup(["script", "style", "table", "footer", "nav", ".mw-editsection", ".reference", ".infobox"]):
                element.decompose()
            
            # Main content area for Wikipedia
            content = soup.find(id="mw-content-text")
            if not content:
                content = soup.find("main")
            
            if not content:
                self.logger.warning(f"Content not found for URL: {url}")
                return None
                
            paragraphs = content.find_all("p")
            text = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            # Clean up text (remove [1], [2] etc and extra whitespace)
            text = re.sub(r'\[\d+\]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error fetching Wikipedia URL {url}: {e}")
            return None

    def fetch_general_text(self, url: str) -> Optional[str]:
        """Generic web text extraction for non-Wikipedia sites."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Standard cleanup
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Try to find main content
            main = soup.find("main") or soup.find("article") or soup.find("body")
            
            if not main:
                return None
                
            # Extract text
            text = main.get_text(separator="\n")
            
            # Clean up
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error fetching URL {url}: {e}")
            return None

    def crawl_urls(self, urls: List[str]) -> List[str]:
        """Fetches text from a list of URLs."""
        results = []
        for url in urls:
            self.logger.info(f"Crawling URL: {url}")
            if "wikipedia.org" in url:
                text = self.fetch_wikipedia_text(url)
            else:
                text = self.fetch_general_text(url)
                
            if text and len(text) > 100: # Ignore very short pages
                results.append(text)
                
        return results
