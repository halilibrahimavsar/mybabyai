"""Web Crawler — fetch clean text from URLs with optional sub-URL discovery."""

import logging
import re
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebCrawler:
    """Fetches and cleans web-page text for training data.

    Supports:
    * Wikipedia articles (special handling)
    * General web pages
    * Sub-URL discovery (internal links, same domain)
    * Recursive crawling with depth control
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    TIMEOUT = 15

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl_urls(self, urls: List[str]) -> List[str]:
        """Fetch text from a flat list of URLs (no sub-link following)."""
        texts: List[str] = []
        for url in urls:
            try:
                text = self._fetch_text(url)
                if text and len(text.strip()) > 100:
                    texts.append(text.strip())
                    logger.info("Fetched %d chars from %s", len(text), url)
            except Exception as e:
                logger.error("Error fetching %s: %s", url, e)
        return texts

    def crawl_with_depth(
        self,
        seed_urls: List[str],
        depth: int = 1,
        max_pages: int = 50,
    ) -> List[str]:
        """Crawl *seed_urls* and follow internal links up to *depth* levels.

        Args:
            seed_urls: Starting URLs.
            depth: How many link-hops to follow (0 = same as ``crawl_urls``).
            max_pages: Safety cap on total pages visited.

        Returns:
            List of cleaned text strings (one per page).
        """
        visited: Set[str] = set()
        texts: List[str] = []
        queue: List[tuple] = [(url, 0) for url in seed_urls]

        while queue and len(visited) < max_pages:
            url, current_depth = queue.pop(0)
            normalized = self._normalize_url(url)

            if normalized in visited:
                continue
            visited.add(normalized)

            try:
                html = self._fetch_html(url)
                if html is None:
                    continue

                text = self._extract_text(html, url)
                if text and len(text.strip()) > 100:
                    texts.append(text.strip())
                    logger.info(
                        "[depth=%d] Fetched %d chars from %s",
                        current_depth,
                        len(text),
                        url,
                    )

                # Discover sub-links if we haven't exceeded depth
                if current_depth < depth:
                    sub_urls = self._discover_sub_urls(html, url)
                    for sub in sub_urls:
                        if self._normalize_url(sub) not in visited:
                            queue.append((sub, current_depth + 1))

            except Exception as e:
                logger.error("Error crawling %s: %s", url, e)

        logger.info(
            "Crawl complete: visited %d pages, extracted %d texts.",
            len(visited),
            len(texts),
        )
        return texts

    def discover_sub_urls(self, url: str, max_links: int = 50) -> List[str]:
        """Public helper: fetch a page and return its internal links."""
        html = self._fetch_html(url)
        if html is None:
            return []
        return self._discover_sub_urls(html, url)[:max_links]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_html(self, url: str) -> Optional[str]:
        """Download raw HTML; returns None on failure."""
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=self.TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            return None

    def _fetch_text(self, url: str) -> str:
        """Download and extract clean text from a single URL."""
        html = self._fetch_html(url)
        if html is None:
            return ""
        return self._extract_text(html, url)

    def _extract_text(self, html: str, url: str) -> str:
        """Parse HTML → clean text, with Wikipedia-specific handling."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "iframe", "noscript"]):
            tag.decompose()

        if "wikipedia.org" in url:
            return self._extract_wikipedia(soup)
        return self._extract_general(soup)

    def _extract_wikipedia(self, soup: BeautifulSoup) -> str:
        """Extract main article body from Wikipedia."""
        content = soup.find("div", {"id": "mw-content-text"})
        if not content:
            content = soup.find("div", {"id": "bodyContent"})
        if not content:
            return soup.get_text(separator="\n", strip=True)

        # Remove edit links, reference sections, navboxes
        for cls in ["mw-editsection", "reflist", "navbox", "sistersitebox",
                     "noprint", "metadata", "mbox-small"]:
            for el in content.find_all(class_=cls):
                el.decompose()
        for el in content.find_all("table", class_="infobox"):
            el.decompose()
        for sup in content.find_all("sup", class_="reference"):
            sup.decompose()

        text = content.get_text(separator="\n", strip=True)
        return self._clean_text(text)

    def _extract_general(self, soup: BeautifulSoup) -> str:
        """Extract main content from any web page."""
        # Try common content containers first
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", {"role": "main"})
            or soup.find("div", class_=re.compile(r"(content|article|post|entry)", re.I))
        )
        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)
        return self._clean_text(text)

    def _discover_sub_urls(self, html: str, base_url: str) -> List[str]:
        """Extract same-domain links from HTML."""
        soup = BeautifulSoup(html, "html.parser") if isinstance(html, str) else html
        base_domain = urlparse(base_url).netloc
        found: List[str] = []
        seen: Set[str] = set()

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            # Skip fragments-only, javascript, mailto
            if href.startswith(("#", "javascript:", "mailto:")):
                continue

            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)

            # Same domain only
            if parsed.netloc != base_domain:
                continue
            # Strip fragment
            clean = parsed._replace(fragment="").geturl()
            norm = self._normalize_url(clean)

            if norm not in seen:
                seen.add(norm)
                found.append(clean)

        return found

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL for dedup (lowercase, strip trailing slash)."""
        parsed = urlparse(url.lower())
        path = parsed.path.rstrip("/")
        return f"{parsed.scheme}://{parsed.netloc}{path}"

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove excessive whitespace and short lines."""
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            if len(line) > 20:  # Skip very short lines (nav remnants)
                cleaned.append(line)
        return "\n".join(cleaned)
