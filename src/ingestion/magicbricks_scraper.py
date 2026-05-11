"""
src/ingestion/magicbricks_scraper.py
MagicBricks Playwright scraper — multi-city residential listings

MagicBricks uses window.SERVER_PRELOADED_STATE_.searchResult (NOT __NEXT_DATA__).

Field mapping (confirmed from live JSON):
  id          → property_id
  priceD      → price string "73.41 Cr"  (use this, not raw `price` int)
  caSqFt      → area sqft (flat/house/independent_floor)
  la          → area sq-yrd (plot) — convert x 9.0 to sqft
  bedroomD    → bhk string "4"
  bathD       → bathroom string "4"
  balconiesD  → balcony string "4"
  furnishedD  → furnish string "Unfurnished"
  acD         → age string "Less than 5 years"
  ty          → property type code: 10000=plot, 10001=house, 10002=flat, 10003=floor
  propTypeD   → property type display: "Apartment", "Residential Plot", etc.
  lmtDName    → locality name
  prjname     → society / sector name
  landmarkDetails → nearby places list ["19202|School Name", ...]
  psmAmenDesc → amenities list ["12202|Convenience|Lift|", ...]
  ltcoordGeo  → "lat,long" string
"""

import asyncio
import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── City config ───────────────────────────────────────────────────────────────

def load_city_config(city: str) -> dict:
    """Load city-specific scraping config from configs/cities.yaml."""
    config_path = Path("configs/cities.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing: {config_path}")
    with open(config_path) as f:
        all_configs = yaml.safe_load(f)
    if city not in all_configs:
        raise ValueError(f"City '{city}' not in cities.yaml. Available: {list(all_configs)}")
    return all_configs[city]


# ── User agents ───────────────────────────────────────────────────────────────

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",

    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",

    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
    "Gecko/20100101 Firefox/121.0",

    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",

    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]


# ── Property type code mapping ────────────────────────────────────────────────

# MagicBricks `ty` field codes (confirmed from live data)
TY_CODE_MAP = {
    "10000": "plot",
    "10001": "house",
    "10002": "flat",
    "10003": "independent_floor",
    "10004": "flat",   # penthouse
    "10005": "flat",   # studio
}


# ── Extraction: SERVER_PRELOADED_STATE_ ───────────────────────────────────────

async def extract_next_data(page) -> list[dict]:
    """
    Primary extraction: read listing data from window.SERVER_PRELOADED_STATE_.

    MagicBricks no longer uses __NEXT_DATA__. All search results are in
    window.SERVER_PRELOADED_STATE_.searchResult (list of dicts).

    Fallback: tries __NEXT_DATA__ for older page formats.
    Returns list of raw property dicts. Empty list if extraction fails.
    """
    try:
        results = await page.evaluate(
            "() => window.SERVER_PRELOADED_STATE_?.searchResult || []"
        )
        if results:
            logger.debug(f"SERVER_PRELOADED_STATE_ extracted {len(results)} raw records")
            return results

        # Fallback: __NEXT_DATA__ (old format)
        raw = await page.evaluate(
            "() => document.getElementById('__NEXT_DATA__')?.textContent"
        )
        if raw:
            data = json.loads(raw)
            results = (
                data.get("props", {})
                    .get("pageProps", {})
                    .get("searchResults", {})
                    .get("results", [])
            )
            if not results:
                results = (
                    data.get("props", {})
                        .get("pageProps", {})
                        .get("data", {})
                        .get("results", [])
                )
            if results:
                logger.debug(f"__NEXT_DATA__ fallback: {len(results)} records")
                return results

        return []

    except Exception as e:
        logger.warning(f"extract_next_data failed: {e}")
        return []


# ── Extraction: DOM fallback ──────────────────────────────────────────────────

async def extract_from_dom(page) -> list[dict]:
    """
    DOM fallback extraction from MagicBricks listing cards.
    Used only when SERVER_PRELOADED_STATE_ is unavailable.
    Note: amenities and nearbylocations will be empty in DOM mode.
    """
    records = []
    try:
        await page.wait_for_selector(".mb-srp__card", timeout=10_000)
        cards = await page.query_selector_all(".mb-srp__card")

        for card in cards:
            try:
                rec = {}

                title_el = await card.query_selector(".mb-srp__card--title")
                if title_el:
                    rec["title"] = (await title_el.inner_text()).strip()

                price_el = await card.query_selector(".mb-srp__card__price--amount")
                if price_el:
                    rec["priceD"] = (await price_el.inner_text()).strip()

                summary_fields = {
                    "super-area":  "caSqFt_raw",
                    "carpet-area": "carpetArea",
                    "bathroom":    "bathD",
                    "balcony":     "balconiesD",
                    "furnishing":  "furnishedD",
                    "society":     "prjname",
                    "status":      "acD",
                    "parking":     "parking_raw",
                }
                for data_key, col_name in summary_fields.items():
                    el = await card.query_selector(
                        f"[data-summary='{data_key}'] .mb-srp__card__summary--value"
                    )
                    if el:
                        rec[col_name] = (await el.inner_text()).strip()

                if "title" in rec:
                    bhk_match = re.search(r"(\d)\s*BHK", rec["title"], re.IGNORECASE)
                    if bhk_match:
                        rec["bedroomD"] = bhk_match.group(1)

                    title_lower = rec["title"].lower()
                    if any(k in title_lower for k in ["plot", "land"]):
                        rec["propTypeD"] = "Residential Plot"
                        rec["ty"] = "10000"
                    elif any(k in title_lower for k in ["villa", "bungalow", "kothi"]):
                        rec["propTypeD"] = "Independent House"
                        rec["ty"] = "10001"
                    elif "builder floor" in title_lower:
                        rec["propTypeD"] = "Builder Floor"
                        rec["ty"] = "10003"
                    else:
                        rec["propTypeD"] = "Apartment"
                        rec["ty"] = "10002"

                    sector_match = re.search(
                        r"Sector[\s\-]*([\w\d]+)", rec["title"], re.IGNORECASE
                    )
                    if sector_match:
                        rec["lmtDName"] = f"Sector {sector_match.group(1)}"

                if rec:
                    records.append(rec)

            except Exception:
                continue

    except Exception as e:
        logger.debug(f"DOM extraction failed: {e}")

    logger.debug(f"DOM fallback extracted {len(records)} records")
    return records


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_price(raw) -> Optional[float]:
    """
    Parse price to float in Crores.

    Prefer priceD string ("73.41 Cr") over raw int (paisa: 734187000).
    Raw int: divide by 10_000_000 to get Crores.
    """
    if raw is None:
        return None

    if isinstance(raw, (int, float)):
        if raw <= 0:
            return None
        return round(raw / 10_000_000, 3)

    raw_str = str(raw).lower().strip()
    if any(kw in raw_str for kw in ["request", "contact", "call", "price on"]):
        return None

    cleaned = raw_str.replace(",", "").replace("\n", "").strip()
    cleaned = re.sub(r"[₹rs\s]", "", cleaned)

    match = re.search(r"(\d+\.?\d*)", cleaned)
    if not match:
        return None
    num = float(match.group(1))

    if "cr" in cleaned or "crore" in cleaned:
        return round(num, 3)
    elif "lac" in cleaned or "lakh" in cleaned:
        return round(num / 100.0, 3)
    elif re.search(r"\d\s*l\b", cleaned):
        return round(num / 100.0, 3)
    elif num >= 100_000:
        return round(num / 10_000_000.0, 3)
    elif num < 1_000:
        return round(num, 3)
    return None


AREA_CONVERSIONS = {
    "sq.mt":   10.7639,
    "sq.m":    10.7639,
    "sqm":     10.7639,
    "sq.yd":   9.0,
    "sqyd":    9.0,
    "sq-yrd":  9.0,
    "yard":    9.0,
    "bigha":   27_225.0,
    "biswa":   1_361.25,
    "acre":    43_560.0,
    "marla":   272.25,
    "kanal":   5_445.0,
}


def parse_area(raw, unit_hint: str = "") -> Optional[float]:
    """
    Parse area to float in sqft.
    unit_hint: pass "sq-yrd" for plots so conversion applies.
    """
    if raw is None:
        return None

    if isinstance(raw, (int, float)):
        if raw <= 0:
            return None
        if unit_hint in AREA_CONVERSIONS:
            return round(raw * AREA_CONVERSIONS[unit_hint], 1)
        return round(float(raw), 1)

    raw_lower = str(raw).lower().replace(",", "").strip()
    if not raw_lower or raw_lower == "nan":
        return None

    range_match = re.search(r"(\d+\.?\d*)\s*[-to]+\s*(\d+\.?\d*)", raw_lower)
    if range_match:
        num = (float(range_match.group(1)) + float(range_match.group(2))) / 2
    else:
        num_match = re.search(r"(\d+\.?\d*)", raw_lower)
        if not num_match:
            return None
        num = float(num_match.group(1))

    if num <= 0:
        return None

    for unit, factor in AREA_CONVERSIONS.items():
        if unit in raw_lower:
            return round(num * factor, 1)

    if unit_hint in AREA_CONVERSIONS:
        return round(num * AREA_CONVERSIONS[unit_hint], 1)

    return round(num, 1)


def parse_floor(raw: str) -> tuple[int, int]:
    """Parse floor string to (floor_position, total_floors)."""
    if not raw:
        return 0, 5
    cleaned = str(raw).replace("\xa0", " ").lower().strip()
    m = re.search(r"(ground|\d+)\s*(?:out of|of|\/)\s*(\d+)", cleaned)
    if m:
        fp = 0 if m.group(1) == "ground" else int(m.group(1))
        total = int(m.group(2))
        return fp, max(total, fp + 1)
    if re.fullmatch(r"(ground|g|gf|ground floor|gr)", cleaned):
        return 0, 1
    if "+" in cleaned:
        num_m = re.search(r"(\d+)", cleaned)
        if num_m:
            return int(num_m.group(1)), 10
    num_m = re.search(r"(\d+)", cleaned)
    if num_m:
        fp = int(num_m.group(1))
        return fp, fp + 2
    return 0, 5


def parse_landmark_details(landmarks: list) -> str:
    """
    Parse landmarkDetails list to comma-separated nearby string.
    Format: "19202|School Name|5 minutes" or "19202|School Name"
    Extracts the place name (index 1).
    """
    if not landmarks:
        return ""
    names = []
    for item in landmarks:
        parts = str(item).split("|")
        if len(parts) >= 2 and parts[1].strip():
            names.append(parts[1].strip())
    return ",".join(names)


def parse_amenity_desc(amenities: list) -> str:
    """
    Parse psmAmenDesc list to comma-separated amenity string.
    Format: "12202|Convenience|Lift|" — extract index 2 (actual name).
    Falls back to index 1 (category) if name is empty.
    """
    if not amenities:
        return ""
    names = []
    for item in amenities:
        parts = str(item).split("|")
        if len(parts) >= 3 and parts[2].strip():
            names.append(parts[2].strip())
        elif len(parts) >= 2 and parts[1].strip():
            names.append(parts[1].strip())
    return ",".join(names)


# ── Property type detection ───────────────────────────────────────────────────

def _detect_property_type(raw: dict) -> str:
    """
    Detect property type from SERVER_PRELOADED_STATE_ record.

    Priority:
      1. ty code (most reliable — MagicBricks backend sets this)
      2. propTypeD display string
      3. title text (DOM fallback only)

    Why plot before house: "Independent Plot" has "independent" keyword
    which would match house if checked first.
    """
    # 1. ty code
    ty = str(raw.get("ty", "")).strip()
    if ty in TY_CODE_MAP:
        return TY_CODE_MAP[ty]

    # 2. propTypeD
    prop_type_d = str(raw.get("propTypeD", "")).lower().strip()
    if any(k in prop_type_d for k in ["plot", "land", "agricultural"]):
        return "plot"
    if any(k in prop_type_d for k in ["house", "villa", "bungalow", "kothi", "independent house"]):
        return "house"
    if any(k in prop_type_d for k in ["builder floor", "floor apartment", "independent floor"]):
        return "independent_floor"

    # 3. title fallback
    title_raw = str(raw.get("title", "")).lower()
    if "plot" in title_raw and "bhk" not in title_raw:
        return "plot"
    if any(k in title_raw for k in ["villa", "bungalow", "kothi"]):
        return "house"
    if "builder floor" in title_raw:
        return "independent_floor"

    return "flat"


# ── Record normalisation ──────────────────────────────────────────────────────

def normalize_record(raw: dict, city: str, seen_ids: set) -> Optional[dict]:
    """
    Convert raw SERVER_PRELOADED_STATE_ record to clean internal schema dict.

    Returns None for duplicate, unparseable price, or unparseable area.
    Plot: area from la field (sq-yrd converted to sqft), bhk=0, floor=Ground.
    """

    # ── Deduplication ─────────────────────────────────────────────────
    prop_id = str(raw.get("id", raw.get("encId", "")))
    if not prop_id or prop_id == "None":
        price_clean = re.sub(r"[^\d.]", "", str(raw.get("priceD", "0")))[:8]
        area_clean  = re.sub(r"[^\d.]", "", str(raw.get("caSqFt", raw.get("la", "0"))))[:6]
        loc_clean   = re.sub(r"[^a-zA-Z0-9]", "", str(raw.get("lmtDName", "x")))[:10]
        prop_id     = f"{city}_{price_clean}_{area_clean}_{loc_clean}"
    if prop_id in seen_ids:
        return None
    seen_ids.add(prop_id)

    # ── Property type ─────────────────────────────────────────────────
    property_type = _detect_property_type(raw)

    # ── Price ─────────────────────────────────────────────────────────
    price = parse_price(raw.get("priceD") or raw.get("price"))
    if price is None:
        return None

    # ── Area ──────────────────────────────────────────────────────────
    if property_type == "plot":
        # la = plot area in sq-yrd
        area = parse_area(
            raw.get("la") or raw.get("caSqFt") or raw.get("ca"),
            unit_hint="sq-yrd"
        )
    else:
        # caSqFt already in sqft
        area = parse_area(raw.get("caSqFt") or raw.get("ca") or raw.get("carpetArea"))

    if area is None:
        return None

    # ── Floor ─────────────────────────────────────────────────────────
    if property_type == "plot":
        floor_str    = "Ground"
        floor_pos    = 0
        total_floors = 1
    else:
        floor_raw = str(raw.get("floorNum", raw.get("floor", "")))
        total_raw = str(raw.get("totalFloor", raw.get("totalFloors", "")))
        if total_raw.isdigit() and floor_raw:
            floor_str = f"{floor_raw} out of {total_raw}"
        else:
            floor_str = floor_raw or "0"
        floor_pos, total_floors = parse_floor(floor_str)

    # ── BHK ───────────────────────────────────────────────────────────
    if property_type == "plot":
        bhk = 0
    else:
        bhk_raw = raw.get("bedroomD", raw.get("bedrooms", raw.get("bedroom", 2)))
        try:
            bhk = int(bhk_raw)
        except (ValueError, TypeError):
            m = re.search(r"(\d+)", str(bhk_raw))
            bhk = int(m.group(1)) if m else 2

    # ── Bathroom ──────────────────────────────────────────────────────
    bath_raw = raw.get("bathD", raw.get("bathroom", raw.get("bathrooms", max(bhk, 1))))
    try:
        bathroom = int(bath_raw)
    except (ValueError, TypeError):
        m = re.search(r"(\d+)", str(bath_raw))
        bathroom = int(m.group(1)) if m else max(bhk, 1)

    # ── Balcony ───────────────────────────────────────────────────────
    bal_raw = raw.get("balconiesD", raw.get("balcony", raw.get("balconies", 0)))
    try:
        balcony = int(bal_raw)
    except (ValueError, TypeError):
        m = re.search(r"(\d+)", str(bal_raw))
        balcony = int(m.group(1)) if m else 0

    # ── Amenities ─────────────────────────────────────────────────────
    amenities_raw = raw.get("psmAmenDesc", raw.get("amenities", raw.get("features", [])))
    amenities_str = (
        parse_amenity_desc(amenities_raw)
        if isinstance(amenities_raw, list)
        else str(amenities_raw) if amenities_raw else ""
    )

    # ── Nearby locations ──────────────────────────────────────────────
    nearby_raw = raw.get("landmarkDetails", raw.get("nearbyLocations", raw.get("nearby", [])))
    nearby_str = (
        parse_landmark_details(nearby_raw)
        if isinstance(nearby_raw, list)
        else str(nearby_raw) if nearby_raw else ""
    )

    # ── Lat / Long ────────────────────────────────────────────────────
    lat, lng = 0.0, 0.0
    geo = str(raw.get("ltcoordGeo", ""))
    if geo and "," in geo:
        try:
            parts = geo.split(",")
            lat   = float(parts[0].strip())
            lng   = float(parts[1].strip())
        except (ValueError, IndexError):
            pass

    # ── Coaching flag (Kota) ──────────────────────────────────────────
    is_near_coaching = 0
    if city == "kota":
        coaching_text    = (nearby_str + " " + str(raw.get("lmtDName", ""))).lower()
        coaching_keywords = ["allen", "resonance", "vedanta", "aakash", "coaching"]
        is_near_coaching  = int(any(kw in coaching_text for kw in coaching_keywords))

    # ── RERA + possession ─────────────────────────────────────────────
    rera_approved = int(bool(raw.get("reraApproved") or raw.get("isReraApproved")))
    possession    = str(raw.get("acD", raw.get("possessionStatus", "1-5 years")))
    furnish_raw   = str(
        raw.get("furnishedD", raw.get("furnishStatus", raw.get("furnishing", "unfurnished")))
    ).lower()

    return {
        "property_id":      prop_id,
        "city":             city,
        "property_type":    property_type,
        "price":            price,
        "area":             area,
        "bhk":              min(max(int(bhk), 0), 10),
        "bathroom":         bathroom,
        "balcony":          balcony,
        "floor":            floor_str,
        "floor_pos":        floor_pos,
        "total_floors":     total_floors,
        "age":              possession,
        "furnish":          furnish_raw,
        "locality":         str(raw.get("lmtDName", raw.get("localityName", "Unknown"))),
        "sector":           str(raw.get("prjname",  raw.get("societyName",  "Unknown"))),
        "amenities":        amenities_str,
        "nearbylocations":  nearby_str,
        "parking":          int(raw.get("parking", 0) or 0),
        "facing":           str(raw.get("facing", "")),
        "rating":           float(raw.get("rating", raw.get("projectRating", 4.0)) or 4.0),
        "lat":              lat,
        "lng":              lng,
        "is_near_coaching": is_near_coaching,
        "rera_approved":    rera_approved,
        "source":           "magicbricks",
        "scraped_at":       datetime.now().isoformat(),
    }


# ── City scraper ──────────────────────────────────────────────────────────────

async def scrape_city(city: str, max_listings: int = 3000, headless: bool = True) -> pd.DataFrame:
    """
    Scrape all listings for a city from MagicBricks.
    Uses SERVER_PRELOADED_STATE_ as primary, DOM as fallback.
    """
    from playwright.async_api import async_playwright

    config   = load_city_config(city)
    base_url = config["base_url"]
    seen_ids = load_seen_ids(city)

    all_records          = []
    consecutive_empty    = 0
    page_num             = 1
    max_empty            = 3
    no_new_records_count = 0
    prev_total           = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={"width": 1366, "height": 768},
        )
        page = await context.new_page()

        while len(all_records) < max_listings:
            url = f"{base_url}&page={page_num}"
            logger.info(f"[{city}] Page {page_num} | collected={len(all_records)}/{max_listings}")

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                await asyncio.sleep(random.uniform(3.0, 5.0))

                raw_records = await extract_next_data(page)
                if not raw_records:
                    raw_records = await extract_from_dom(page)

                if not raw_records:
                    consecutive_empty += 1
                    logger.warning(
                        f"[{city}] Empty page {page_num} ({consecutive_empty}/{max_empty})"
                    )
                    if consecutive_empty >= max_empty:
                        logger.info(f"[{city}] Max empty pages hit, stopping.")
                        break
                    page_num += 1
                    continue

                consecutive_empty = 0
                added_this_page   = 0

                for rec in raw_records:
                    cleaned = normalize_record(rec, city, seen_ids)
                    if cleaned:
                        all_records.append(cleaned)
                        added_this_page += 1

                logger.info(
                    f"[{city}] Page {page_num} -> {len(raw_records)} raw, "
                    f"{added_this_page} new, {len(all_records)} total"
                )

                current_total = len(all_records)
                if current_total == prev_total:
                    no_new_records_count += 1
                    logger.warning(
                        f"[{city}] No new records for {no_new_records_count} pages"
                    )
                    if no_new_records_count >= 5:
                        logger.info(f"[{city}] Listings exhausted at {current_total}. Stopping.")
                        break
                else:
                    no_new_records_count = 0

                prev_total = current_total
                page_num  += 1

            except Exception as e:
                logger.error(f"[{city}] Page {page_num} error: {e}")
                consecutive_empty += 1
                if consecutive_empty >= max_empty:
                    logger.info(f"[{city}] Too many errors, stopping.")
                    break
                page_num += 1
                await asyncio.sleep(5)

        await browser.close()

    df = pd.DataFrame(all_records)
    logger.info(f"[{city}] Done. {len(df)} records scraped.")
    if len(df) > 0:
        logger.info(f"[{city}] Property types:\n{df['property_type'].value_counts().to_string()}")
        logger.info(f"[{city}] Amenities non-empty:  {(df['amenities'] != '').sum()}/{len(df)}")
        logger.info(f"[{city}] Nearby non-empty:     {(df['nearbylocations'] != '').sum()}/{len(df)}")
    return df


# ── Persistence ───────────────────────────────────────────────────────────────

def save_raw(df: pd.DataFrame, city: str) -> str:
    """Save scraped DataFrame as date-stamped parquet."""
    today    = datetime.now().strftime("%Y%m%d")
    out_dir  = Path(f"data/raw/{city}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"magicbricks_{city}_{today}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} records to {out_path}")
    return str(out_path)


def load_seen_ids(city: str) -> set:
    """Load already-scraped property IDs to prevent duplicates across runs."""
    seen    = set()
    raw_dir = Path(f"data/raw/{city}")
    if not raw_dir.exists():
        return seen
    for f in raw_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(f, columns=["property_id"])
            seen.update(df["property_id"].astype(str).tolist())
        except Exception:
            continue
    logger.info(f"[{city}] Loaded {len(seen)} seen IDs from existing parquets")
    return seen


# ── Quick test ────────────────────────────────────────────────────────────────

async def _test_extraction():
    """Quick manual test — scrape 50 listings from Gurgaon."""
    df = await scrape_city("gurgaon", max_listings=50, headless=False)

    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nProperty types:\n{df['property_type'].value_counts()}")
    print(f"\nAmenities sample:\n{df['amenities'].head(3).tolist()}")
    print(f"\nNearby sample:\n{df['nearbylocations'].head(3).tolist()}")
    print(f"\nSample row (flat):\n{df[df['property_type']=='flat'].iloc[0].to_dict()}")
    print(f"\nNull counts:\n{df.isnull().sum()}")

    path = save_raw(df, "gurgaon")
    logger.info(f"Saved to {path}")


if __name__ == "__main__":
    asyncio.run(_test_extraction())