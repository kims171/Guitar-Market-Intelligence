"""
scraper.py — Reverb.com Sold Listings Scraper
Collects historical sold guitar listings and saves them to raw_listings.csv
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://reverb.com/marketplace"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

TARGET_BRANDS = ["Fender", "Gibson", "Gretsch", "Rickenbacker", "Epiphone"]

# Reverb category slug for electric guitars
CATEGORY_SLUG = "electric-guitars"

OUTPUT_PATH = Path("data/raw_listings.csv")


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

def build_search_url(brand: str, page: int = 1) -> str:
    """Build a Reverb sold-listings search URL for a given brand."""
    return (
        f"{BASE_URL}?query={brand}+vintage+electric+guitar"
        f"&category={CATEGORY_SLUG}"
        f"&decade%5B%5D=1950s&decade%5B%5D=1960s"
        f"&decade%5B%5D=1970s&decade%5B%5D=1980s"
        f"&sold_listings=true"
        f"&page={page}"
    )


def parse_listing_card(card) -> dict | None:
    """
    Extract fields from a single Reverb listing <article> card.
    Returns None if essential fields are missing.
    """
    try:
        title_el = card.select_one("[data-listing-id] .listing-card__title")
        price_el = card.select_one(".price-display__price")
        condition_el = card.select_one(".listing-card__condition")
        sold_date_el = card.select_one(".listing-card__nudge--sold")
        link_el = card.select_one("a.listing-card__title")

        if not (title_el and price_el):
            return None

        price_str = price_el.get_text(strip=True).replace("$", "").replace(",", "")
        price = float(price_str) if price_str else None

        return {
            "title": title_el.get_text(strip=True),
            "price_usd": price,
            "condition": condition_el.get_text(strip=True) if condition_el else "Unknown",
            "sold_date": sold_date_el.get_text(strip=True) if sold_date_el else None,
            "url": "https://reverb.com" + link_el["href"] if link_el else None,
            "scraped_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.debug(f"Failed to parse card: {e}")
        return None


def scrape_listing_description(url: str, session: requests.Session) -> str:
    """Fetch the full description text from an individual listing page."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        desc_el = soup.select_one(".listing-description")
        return desc_el.get_text(separator=" ", strip=True) if desc_el else ""
    except Exception as e:
        logger.debug(f"Could not fetch description from {url}: {e}")
        return ""


def scrape_brand(brand: str, max_pages: int = 5) -> list[dict]:
    """Scrape sold listings for a single brand across multiple pages."""
    session = requests.Session()
    results = []

    for page in range(1, max_pages + 1):
        url = build_search_url(brand, page)
        logger.info(f"Scraping {brand} — page {page}: {url}")

        try:
            resp = session.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Request failed: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        cards = soup.select("article.listing-card")

        if not cards:
            logger.info(f"No more listings found for {brand} on page {page}.")
            break

        for card in cards:
            listing = parse_listing_card(card)
            if listing is None:
                continue

            # Politely fetch the description from the individual page
            if listing["url"]:
                listing["description"] = scrape_listing_description(
                    listing["url"], session
                )
                time.sleep(random.uniform(1.5, 3.0))  # Respectful crawl delay
            else:
                listing["description"] = ""

            listing["brand"] = brand
            results.append(listing)

        logger.info(f"  → {len(cards)} cards found, {len(results)} total so far.")

        # Page-level delay
        time.sleep(random.uniform(2.0, 4.0))

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_scraper(brands: list[str] = TARGET_BRANDS, max_pages: int = 5):
    """Scrape all target brands and save results to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_listings = []

    for brand in brands:
        brand_listings = scrape_brand(brand, max_pages=max_pages)
        all_listings.extend(brand_listings)
        logger.info(f"Finished {brand}: {len(brand_listings)} listings collected.")

    df = pd.DataFrame(all_listings)
    df.drop_duplicates(subset=["url"], inplace=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved {len(df)} listings to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    run_scraper()
