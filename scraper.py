"""
scraper.py — Reverb Official API Data Collector
Uses api.reverb.com (no scraping, no bot detection, clean JSON).

SETUP (one-time):
  1. Create a free Reverb account at reverb.com
  2. Go to reverb.com/my/account/apps → "Personal Access Tokens"
  3. Create a token with scope: "Read listings"
  4. Create a file called .env in this folder and add:
       REVERB_TOKEN=your_token_here
  OR just paste your token into REVERB_TOKEN below.

Run:
    python scraper.py
"""

import os
import time
import logging
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG — paste your token here if you don't want to use .env
# ---------------------------------------------------------------------------
REVERB_TOKEN = os.getenv("REVERB_TOKEN", "aa76ffb0b99a37526d766dcc59a59001ccb167d1da580dbd749e6689d31d175e")

BASE_URL = "https://api.reverb.com/api"
HEADERS = {
    "Content-Type": "application/hal+json",
    "Accept": "application/hal+json",
    "Accept-Version": "3.0",
    "Authorization": f"Bearer {REVERB_TOKEN}",
}

TARGET_BRANDS = ["Fender", "Gibson", "Gretsch", "Rickenbacker", "Epiphone"]
DECADES = ["1950s", "1960s", "1970s", "1980s"]
OUTPUT_PATH = Path("data/raw_listings.csv")
PER_PAGE = 50        # Max per page Reverb allows
MAX_PAGES = 10       # 10 pages × 50 = 500 listings per brand


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def reverb_get(endpoint: str, params: dict) -> dict | None:
    """Make a GET request to the Reverb API. Returns JSON or None on failure."""
    url = f"{BASE_URL}/{endpoint}"
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as e:
        if resp.status_code == 401:
            logger.error("401 Unauthorized — check your REVERB_TOKEN in .env")
        elif resp.status_code == 429:
            logger.warning("Rate limited — sleeping 60s")
            time.sleep(60)
        else:
            logger.warning(f"HTTP {resp.status_code} for {url}: {e}")
    except requests.RequestException as e:
        logger.warning(f"Request failed: {e}")
    return None


def flatten_listing(item: dict, brand: str) -> dict:
    """
    Pull the fields we care about from a Reverb listing JSON object.
    Reverb returns nested dicts — we flatten them into a single row.
    """
    price = item.get("price", {})
    condition = item.get("condition", {})
    shipping = item.get("shipping", {})

    return {
        "id":                  item.get("id"),
        "brand":               brand,
        "title":               item.get("title", ""),
        "make":                item.get("make", ""),
        "model":               item.get("model", ""),
        "year":                item.get("year", ""),
        "finish":              item.get("finish", ""),
        "description":         item.get("description", ""),
        "price_amount":        price.get("amount"),
        "price_currency":      price.get("currency", "USD"),
        "condition":           condition.get("display_name", ""),
        "condition_slug":      condition.get("slug", ""),
        "state":               item.get("state"),
        "has_inventory":       item.get("has_inventory"),
        "created_at":          item.get("created_at"),
        "published_at":        item.get("published_at"),
        "url":                 item.get("_links", {}).get("web", {}).get("href", ""),
        "photos":              len(item.get("photos", [])),
        "seller_country":      item.get("seller_country", ""),
        "shop_name":           item.get("shop", {}).get("name", "") if item.get("shop") else "",
    }


# ---------------------------------------------------------------------------
# Collection logic
# ---------------------------------------------------------------------------

def collect_brand_decade(brand: str, decade: str, max_pages: int = MAX_PAGES) -> list[dict]:
    """
    Fetch sold listings for one brand+decade combination via the Reverb API.
    Reverb paginates via _links.next.href — we follow that chain.
    """
    results = []

    params = {
        "query": f"{brand} vintage electric guitar",
        "make":  brand,
        "decade[]": decade,
        "state": "sold",        # sold listings only
        "per_page": PER_PAGE,
        "page": 1,
    }

    for page in range(1, max_pages + 1):
        params["page"] = page
        logger.info(f"  {brand} / {decade} — page {page}")
        data = reverb_get("listings", params)

        if data is None:
            break

        listings = data.get("listings", [])
        if not listings:
            logger.info(f"  No more listings for {brand}/{decade} on page {page}.")
            break

        for item in listings:
            row = flatten_listing(item, brand)
            results.append(row)

        # Respectful delay between pages
        time.sleep(1.0)

        # If Reverb says there's no next page, stop
        links = data.get("_links", {})
        if "next" not in links:
            break

    return results


def collect_all(brands=TARGET_BRANDS, decades=DECADES) -> pd.DataFrame:
    """Collect sold listings across all brands and decades."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if REVERB_TOKEN == "YOUR_TOKEN_HERE":
        logger.error(
            "\n\n========================================================\n"
            "  REVERB_TOKEN not set!\n"
            "  1. Go to reverb.com → My Account → Apps → Personal Access Tokens\n"
            "  2. Create a token with 'Read listings' scope\n"
            "  3. Add REVERB_TOKEN=your_token to a .env file in this folder\n"
            "========================================================"
        )
        return pd.DataFrame()

    all_rows = []
    for brand in brands:
        brand_rows = []
        for decade in decades:
            rows = collect_brand_decade(brand, decade)
            brand_rows.extend(rows)
            time.sleep(2.0)   # Pause between decade queries

        logger.info(f"Collected {len(brand_rows)} sold listings for {brand}.")
        all_rows.extend(brand_rows)

    df = pd.DataFrame(all_rows)

    if df.empty:
        logger.warning("No listings collected. Check your token and try again.")
        return df

    # Deduplicate by Reverb listing ID
    df.drop_duplicates(subset=["id"], inplace=True)

    # Convert price to float
    df["price_usd"] = pd.to_numeric(df["price_amount"], errors="coerce")

    # Rough CAD → USD conversion for any CAD listings
    cad_mask = df["price_currency"].str.upper() == "CAD"
    df.loc[cad_mask, "price_usd"] = df.loc[cad_mask, "price_usd"] * 0.74

    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved {len(df)} listings → {OUTPUT_PATH}")
    return df


# ---------------------------------------------------------------------------
# Also collect active (non-sold) listings for current market context
# ---------------------------------------------------------------------------

def collect_active_listings(brands=TARGET_BRANDS, decades=DECADES, max_pages=5) -> pd.DataFrame:
    """
    Collect currently active (unsold) listings for each brand.
    Useful as a baseline for the 'current asking price' vs predicted price comparison.
    """
    active_path = Path("data/active_listings.csv")
    all_rows = []

    for brand in brands:
        for decade in decades:
            params = {
                "query":    f"{brand} vintage electric guitar",
                "make":     brand,
                "decade[]": decade,
                "per_page": PER_PAGE,
                "page":     1,
            }
            for page in range(1, max_pages + 1):
                params["page"] = page
                data = reverb_get("listings", params)
                if not data or not data.get("listings"):
                    break
                for item in data["listings"]:
                    row = flatten_listing(item, brand)
                    all_rows.append(row)
                time.sleep(1.0)
                if "next" not in data.get("_links", {}):
                    break
            time.sleep(2.0)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.drop_duplicates(subset=["id"], inplace=True)
        df["price_usd"] = pd.to_numeric(df["price_amount"], errors="coerce")
        df.to_csv(active_path, index=False)
        logger.info(f"Saved {len(df)} active listings → {active_path}")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=== Collecting SOLD listings (training data) ===")
    sold_df = collect_all()

    if not sold_df.empty:
        logger.info("\n=== Collecting ACTIVE listings (market context) ===")
        collect_active_listings()

        print(f"\nDone! Summary:")
        print(f"  Sold listings:  {len(sold_df)}")
        print(f"  Brands:         {sold_df['brand'].value_counts().to_dict()}")
        print(f"  Price range:    ${sold_df['price_usd'].min():,.0f} – ${sold_df['price_usd'].max():,.0f}")
        print(f"\nNext step: python preprocess.py")
