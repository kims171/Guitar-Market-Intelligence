"""
preprocess.py — Data Cleaning & Feature Engineering Pipeline
Converts raw_listings.csv → processed_listings.csv ready for model training.

Steps:
  1. Parse year-of-manufacture from title
  2. NLP originality scoring (keyword scanning)
  3. Z-score outlier removal per brand/model
  4. Currency standardization (all prices → USD at historical rate)
  5. Condition normalization (adds "Player Grade" cluster)
  6. Save processed dataset
"""

import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw_listings.csv")
PROCESSED_PATH = Path("data/processed_listings.csv")

# ---------------------------------------------------------------------------
# 1. Year Extraction
# ---------------------------------------------------------------------------

YEAR_PATTERN = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")


def extract_year(title: str) -> int | None:
    """Pull the first 4-digit year (1950–2029) from a listing title."""
    match = YEAR_PATTERN.search(str(title))
    return int(match.group(0)) if match else None


# ---------------------------------------------------------------------------
# 2. NLP Originality Scoring
# ---------------------------------------------------------------------------

# Words that suggest the guitar is NOT all-original
NON_ORIGINAL_KEYWORDS = [
    r"\bnon[- ]?original\b",
    r"\bmodified\b",
    r"\bmodded\b",
    r"\breplaced?\b",
    r"\baftermarket\b",
    r"\brefret(ted)?\b",
    r"\bnon[- ]?stock\b",
    r"\bcustom\s+pickup\b",
    r"\bswapped?\b",
    r"\bnot\s+original\b",
    r"\brewind\b",
    r"\brepair\b",
]

# Words that confirm all-original status
ORIGINAL_KEYWORDS = [
    r"\ball[- ]?original\b",
    r"\b100%\s+original\b",
    r"\bstock\b",
    r"\bunmodified\b",
    r"\bOHSC\b",        # Original Hard Shell Case — strong provenance signal
    r"\boriginal\s+case\b",
    r"\boriginal\s+pickups?\b",
]

PLAYER_GRADE_KEYWORDS = [
    r"\bplayer[- ]?grade\b",
    r"\bplayer[- ]?condition\b",
    r"\bheavy\s+relic\b",
    r"\bbeat\s+up\b",
    r"\bworkhorse\b",
]


def score_originality(description: str) -> int:
    """
    Returns:
        1  → confirmed all-original
        0  → unknown / ambiguous
       -1  → confirmed modified / non-original
    """
    text = str(description).lower()

    # Negation check: "not original" → non-original flag
    if any(re.search(kw, text) for kw in NON_ORIGINAL_KEYWORDS):
        return -1
    if any(re.search(kw, text) for kw in ORIGINAL_KEYWORDS):
        return 1
    return 0


def is_player_grade(description: str) -> bool:
    text = str(description).lower()
    return any(re.search(kw, text) for kw in PLAYER_GRADE_KEYWORDS)


# ---------------------------------------------------------------------------
# 3. Condition Normalization
# ---------------------------------------------------------------------------

CONDITION_MAP = {
    "mint": "Mint",
    "excellent": "Excellent",
    "very good": "Very Good",
    "good": "Good",
    "fair": "Fair",
    "poor": "Poor",
    "non functioning": "Non Functioning",
}


def normalize_condition(raw_condition: str, description: str) -> str:
    """Map raw condition strings → standardized labels, with Player Grade cluster."""
    if is_player_grade(description):
        return "Player Grade"

    lower = str(raw_condition).lower().strip()
    for key, label in CONDITION_MAP.items():
        if key in lower:
            return label
    return "Unknown"


# ---------------------------------------------------------------------------
# 4. Outlier Removal (Z-Score per brand/year group)
# ---------------------------------------------------------------------------

def remove_outliers(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Drop rows whose price_usd is more than `z_threshold` std deviations
    from the mean within their (brand, year_of_manufacture) group.
    This handles celebrity-sale price spikes.
    """
    def group_filter(group):
        if len(group) < 5:          # Too few samples to compute reliable stats
            return group
        z_scores = np.abs(stats.zscore(group["price_usd"].dropna()))
        return group[z_scores < z_threshold]

    before = len(df)
    df = df.groupby(["brand", "year_of_manufacture"], group_keys=False).apply(group_filter)
    after = len(df)
    logger.info(f"Outlier removal: dropped {before - after} rows ({before} → {after})")
    return df


# ---------------------------------------------------------------------------
# 5. Currency Standardization
# ---------------------------------------------------------------------------

# Simplified static CAD→USD rate lookup by year.
# In production, pull from an FX API (e.g. exchangerate.host historical endpoint).
CAD_TO_USD_BY_YEAR = {
    2018: 0.772, 2019: 0.754, 2020: 0.746,
    2021: 0.798, 2022: 0.770, 2023: 0.741,
    2024: 0.733, 2025: 0.720,
}
DEFAULT_FX = 0.74


def standardize_currency(row: pd.Series) -> float:
    """Convert CAD prices to USD using historical FX. Assumes USD if no CAD marker."""
    price = row.get("price_usd", np.nan)
    if pd.isna(price):
        return np.nan

    currency = str(row.get("currency", "USD")).upper()
    if currency == "CAD":
        year = row.get("sale_year")
        rate = CAD_TO_USD_BY_YEAR.get(year, DEFAULT_FX)
        return round(price * rate, 2)

    return round(price, 2)


# ---------------------------------------------------------------------------
# 6. Feature Engineering
# ---------------------------------------------------------------------------

def extract_model_from_title(title: str, brand: str) -> str:
    """Best-effort model extraction: strip the brand and year, return remainder."""
    text = str(title)
    text = re.sub(brand, "", text, flags=re.IGNORECASE)
    text = YEAR_PATTERN.sub("", text)
    return text.strip()


def pickup_config(description: str) -> str:
    """Detect pickup configuration from description (SSS, HH, HSS, etc.)."""
    text = str(description).upper()
    for config in ["HSH", "HSS", "HH", "SSS", "SS", "HS"]:
        if config in text:
            return config
    return "Unknown"


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(raw_path: Path = RAW_PATH, out_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} rows.")

    # --- Parse sale year from sold_date ---
    df["sale_year"] = pd.to_datetime(df["sold_date"], errors="coerce").dt.year

    # --- Year of manufacture ---
    df["year_of_manufacture"] = df["title"].apply(extract_year)

    # --- Guitar model ---
    df["model"] = df.apply(
        lambda r: extract_model_from_title(r["title"], r.get("brand", "")), axis=1
    )

    # --- Originality score ---
    df["originality_score"] = df["description"].apply(score_originality)

    # --- Player grade flag ---
    df["is_player_grade"] = df["description"].apply(is_player_grade).astype(int)

    # --- Condition normalization ---
    df["condition_normalized"] = df.apply(
        lambda r: normalize_condition(r.get("condition", ""), r.get("description", "")), axis=1
    )

    # --- Pickup config ---
    df["pickup_config"] = df["description"].apply(pickup_config)

    # --- Currency standardization ---
    if "currency" not in df.columns:
        df["currency"] = "USD"
    df["price_usd_normalized"] = df.apply(standardize_currency, axis=1)

    # --- Drop rows with no price or manufacture year ---
    df.dropna(subset=["price_usd_normalized", "year_of_manufacture"], inplace=True)
    df["year_of_manufacture"] = df["year_of_manufacture"].astype(int)

    # --- Outlier removal ---
    df = remove_outliers(df)

    # --- One-hot encode categoricals ---
    df = pd.get_dummies(df, columns=["brand", "condition_normalized", "pickup_config"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} processed rows to {out_path}")
    return df


if __name__ == "__main__":
    run_preprocessing()
