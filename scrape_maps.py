#!/usr/bin/env python3
"""
scrape_maps.py
---------------
NEW FILE — Phase 6A extension (bulk data acquisition from MAPS corpus).

What it does
------------
Reads the MAPS Policies Dataset (441,626 privacy policy URLs) and scrapes
a configurable subset into plain text files suitable for chunking.

The MAPS dataset is a zip of URLs — no text included. You must fetch the
actual web pages. This script:
  1. Reads the MAPS URL list
  2. Samples N URLs (default 500 for manageable scale)
  3. Fetches each page with requests + BeautifulSoup
  4. Extracts main text (removes nav, ads, boilerplate)
  5. Saves as .txt files in data/external/maps_policies/

GPU requirement: NONE. Pure network I/O + HTML parsing.

IMPORTANT ETHICAL NOTES:
  - Respect robots.txt and rate-limit (default: 1 req/sec)
  - Many URLs will be dead (404, domain expired) — this is expected
  - Some pages will be login-walled or JS-rendered — we skip those
  - Target: ~30-50% success rate → 150-250 usable policies from 500 URLs

Output:
  data/external/maps_policies/*.txt — one file per successfully scraped policy
  data/external/maps_scrape_log.json — metadata (URL, status, word count)

Usage:
  # Scrape 500 URLs (takes ~10-15 minutes with 1 req/sec)
  python src/chunk_index/scrape_maps.py       --maps_urls data/external/MAPS_Policies_Dataset_v1.0.csv       --out_dir data/external/maps_policies       --max_urls 500       --delay 1.0

  # Then add to your corpus:
  python src/chunk_index/build_chunk_index.py  # will pick up new .txt files
"""

import argparse
import csv
import json
import logging
import os
import random
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "LexGuard-Research/1.0 (Academic privacy policy analysis)"
}


def fetch_policy_text(url: str, timeout: int = 15) -> str:
    """Fetch a privacy policy URL and extract clean text."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "lxml")

        # Remove script, style, nav, footer, header elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Try to find main content area
        main = soup.find("main") or soup.find("article") or soup.find("div", class_="content")
        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n\n".join(lines)

        # Filter out non-policy pages (too short, no privacy keywords)
        if len(text) < 500:
            return ""
        privacy_keywords = ["privacy", "personal data", "information we collect", "cookies"]
        if not any(kw in text.lower() for kw in privacy_keywords):
            return ""

        return text
    except Exception as e:
        logger.debug("Failed to fetch %s: %s", url, e)
        return ""


def scrape_maps(urls_path: str, out_dir: str, max_urls: int, delay: float):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Read URLs
    urls = []
    with open(urls_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                urls.append(row[0].strip())

    logger.info("Loaded %d URLs from %s", len(urls), urls_path)

    # Sample
    if max_urls and len(urls) > max_urls:
        random.seed(42)  # Reproducible
        urls = random.sample(urls, max_urls)
        logger.info("Sampled %d URLs for scraping", len(urls))

    log_entries = []
    success_count = 0
    fail_count = 0

    for i, url in enumerate(urls, 1):
        logger.info("[%d/%d] Fetching %s ...", i, len(urls), url[:80])
        text = fetch_policy_text(url)

        if text:
            # Sanitize filename from URL
            fname = url.replace("https://", "").replace("http://", "").replace("/", "_")[:100]
            fname = fname.replace(".", "_") + ".txt"
            file_path = out_path / fname

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

            word_count = len(text.split())
            log_entries.append({"url": url, "status": "success", "words": word_count, "file": str(file_path)})
            success_count += 1
            logger.info("  -> Saved %s (%d words)", fname, word_count)
        else:
            log_entries.append({"url": url, "status": "failed", "words": 0, "file": None})
            fail_count += 1

        time.sleep(delay)

    # Save log
    log_path = out_path.parent / "maps_scrape_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=2)

    logger.info("=" * 60)
    logger.info("Scrape complete: %d success, %d failed, %d total", success_count, fail_count, len(urls))
    logger.info("Output: %s", out_dir)
    logger.info("Log: %s", log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape privacy policies from MAPS dataset.")
    parser.add_argument("--maps_urls", required=True, help="Path to MAPS CSV/TXT with URLs")
    parser.add_argument("--out_dir", default="data/external/maps_policies", help="Output directory")
    parser.add_argument("--max_urls", type=int, default=500, help="Max URLs to scrape")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests")
    args = parser.parse_args()

    scrape_maps(args.maps_urls, args.out_dir, args.max_urls, args.delay)
