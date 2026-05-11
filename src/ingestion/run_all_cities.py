"""
src/ingestion/run_all_cities.py
PropML — Orchestrates MagicBricks scraping for all 13 cities sequentially.

Phase 3 updates:
  - max_listings read from params.yaml (scraping.max_listings_per_city)
  - Retry logic: failed cities retried up to MAX_RETRIES times
  - Summary report saved to reports/scraping/scrape_summary_YYYYMMDD.json
  - --cities CLI arg: scrape only specific cities
  - --headless CLI arg: disable headless for debugging
  - --dry-run CLI arg: print config without scraping

Run:
  python src/ingestion/run_all_cities.py
  python src/ingestion/run_all_cities.py --cities gurgaon noida
  python src/ingestion/run_all_cities.py --headless false
  python src/ingestion/run_all_cities.py --dry-run

DVC stage (dvc.yaml):
  scrape:
    cmd: python src/ingestion/run_all_cities.py
    deps: [src/ingestion/magicbricks_scraper.py, configs/cities.yaml]
    outs: [data/raw/]
    params: [params.yaml: scraping.max_listings_per_city]
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

ALL_CITIES = [
    "gurgaon", "noida", "chandigarh", "kota",
    "delhi", "mumbai", "bangalore", "chennai",
    "pune", "hyderabad", "ahmedabad", "jaipur", "indore",
]

# Fallback limits if params.yaml is missing
DEFAULT_MAX_PER_CITY = 2000

# Seconds to wait between cities (avoids IP rate limiting)
GAP_BETWEEN_CITIES_SECONDS = 30

# Retry config
MAX_RETRIES    = 2
RETRY_WAIT_SEC = 60


def load_params(params_path: str = "params.yaml") -> dict:
    """Load scraping params from params.yaml. Returns defaults if file missing."""
    defaults = {
        "scraping": {
            "max_listings_per_city": DEFAULT_MAX_PER_CITY,
            "headless":              True,
            "delay_min":             2.0,
            "delay_max":             4.5,
        }
    }
    if not Path(params_path).exists():
        logger.warning(f"params.yaml not found — using defaults")
        return defaults

    with open(params_path) as f:
        params = yaml.safe_load(f)

    merged = defaults.copy()
    merged["scraping"].update(params.get("scraping", {}))
    return merged


# ── Single city scraper ───────────────────────────────────────────────────────

def run_city(city: str, max_listings: int, headless: bool = True) -> dict:
    """
    Scrape one city. Returns result dict with status, records, path, elapsed.

    Imports scraper inside function so this file can be imported without
    triggering Playwright initialization.
    """
    sys.path.insert(0, ".")
    from src.ingestion.magicbricks_scraper import scrape_city, save_raw

    logger.info(f"\n{'='*50}")
    logger.info(f"Starting: {city.upper()} (max {max_listings} listings)")
    start = time.time()

    try:
        df = asyncio.run(
            scrape_city(city, max_listings=max_listings, headless=headless)
        )

        if df.empty:
            return {
                "city":    city,
                "status":  "failed",
                "error":   "0 records scraped",
                "records": 0,
            }

        path    = save_raw(df, city)
        elapsed = round(time.time() - start, 1)

        # Basic quality stats
        amenity_filled = int((df["amenities"] != "").sum()) if "amenities" in df.columns else 0
        nearby_filled  = int((df["nearbylocations"] != "").sum()) if "nearbylocations" in df.columns else 0
        lat_filled     = int((df["lat"] != 0.0).sum()) if "lat" in df.columns else 0
        prop_types     = df["property_type"].value_counts().to_dict() if "property_type" in df.columns else {}

        return {
            "city":            city,
            "status":          "success",
            "records":         len(df),
            "path":            path,
            "elapsed_s":       elapsed,
            "property_types":  prop_types,
            "amenity_filled":  amenity_filled,
            "nearby_filled":   nearby_filled,
            "lat_filled":      lat_filled,
        }

    except Exception as e:
        elapsed = round(time.time() - start, 1)
        logger.error(f"Scraping failed for {city}: {e}")
        return {
            "city":      city,
            "status":    "failed",
            "error":     str(e),
            "records":   0,
            "elapsed_s": elapsed,
        }


def run_city_with_retry(
    city: str,
    max_listings: int,
    headless: bool = True,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """
    Run scraper for a city with retry logic.

    On failure, waits RETRY_WAIT_SEC seconds and retries up to max_retries times.
    Returns last result (success or final failure).
    """
    result = run_city(city, max_listings, headless)

    if result["status"] == "success":
        return result

    for attempt in range(1, max_retries + 1):
        logger.warning(
            f"  [{city}] Attempt {attempt}/{max_retries} failed. "
            f"Retrying in {RETRY_WAIT_SEC}s..."
        )
        time.sleep(RETRY_WAIT_SEC)

        result = run_city(city, max_listings, headless)
        result["retry_attempt"] = attempt

        if result["status"] == "success":
            logger.info(f"  [{city}] Retry {attempt} succeeded.")
            return result

    logger.error(f"  [{city}] All {max_retries} retries failed.")
    return result


# ── Summary report ────────────────────────────────────────────────────────────

def save_summary_report(results: list, total_elapsed: float) -> str:
    """
    Save scraping summary report to reports/scraping/scrape_summary_YYYYMMDD.json.

    Report contains:
    - Per-city: status, records, path, elapsed, property_types, fill rates
    - Total: records, cities scraped, cities failed, elapsed

    Used for:
    - Debugging data quality issues
    - Tracking scraping performance over time
    - CI/CD pipeline pass/fail decisions
    """
    report_dir = Path("reports/scraping")
    report_dir.mkdir(parents=True, exist_ok=True)

    today    = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"scrape_summary_{today}.json"
    path     = report_dir / filename

    total_records  = sum(r["records"] for r in results)
    success_cities = [r["city"] for r in results if r["status"] == "success"]
    failed_cities  = [r["city"] for r in results if r["status"] == "failed"]

    report = {
        "scraped_at":     datetime.now().isoformat(),
        "total_records":  total_records,
        "total_elapsed_s": round(total_elapsed, 1),
        "cities_success": success_cities,
        "cities_failed":  failed_cities,
        "n_success":      len(success_cities),
        "n_failed":       len(failed_cities),
        "results":        results,
    }

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Summary report saved -> {path}")
    return str(path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PropML — Scrape all cities")
    parser.add_argument(
        "--cities", nargs="+", default=None,
        help=f"Cities to scrape (default: all 13). Options: {ALL_CITIES}",
    )
    parser.add_argument(
        "--headless", type=str, default="true",
        help="Run browser headless (default: true). Pass 'false' to see browser.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config without scraping.",
    )
    parser.add_argument(
        "--params", default="params.yaml",
        help="Path to params.yaml (default: params.yaml)",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────
    params      = load_params(args.params)
    max_listing = params["scraping"]["max_listings_per_city"]
    headless    = args.headless.lower() != "false"
    cities      = args.cities if args.cities else ALL_CITIES

    # Validate city names
    invalid = [c for c in cities if c not in ALL_CITIES]
    if invalid:
        logger.error(f"Unknown cities: {invalid}. Valid: {ALL_CITIES}")
        sys.exit(1)

    logger.info("PropML — run_all_cities.py")
    logger.info(f"Cities:       {cities}")
    logger.info(f"Max/city:     {max_listing}")
    logger.info(f"Headless:     {headless}")
    logger.info(f"Retry:        {MAX_RETRIES} attempts")
    logger.info(f"Gap:          {GAP_BETWEEN_CITIES_SECONDS}s between cities")

    if args.dry_run:
        logger.info("DRY RUN — no scraping performed.")
        return

    # ── Scrape all cities ─────────────────────────────────────────────
    results     = []
    total_start = time.time()

    for i, city in enumerate(cities):
        result = run_city_with_retry(
            city,
            max_listings=max_listing,
            headless=headless,
        )
        results.append(result)

        if result["status"] == "success":
            ptypes = result.get("property_types", {})
            logger.info(
                f"  OK   {city}: {result['records']} records "
                f"in {result['elapsed_s']}s | types={ptypes}"
            )
            logger.info(
                f"       amenity_filled={result.get('amenity_filled', 0)}  "
                f"nearby_filled={result.get('nearby_filled', 0)}  "
                f"lat_filled={result.get('lat_filled', 0)}"
            )
        else:
            logger.error(
                f"  FAIL {city}: FAILED — {result.get('error', 'unknown')}"
            )

        # Gap between cities (skip after last)
        if i < len(cities) - 1:
            logger.info(
                f"Waiting {GAP_BETWEEN_CITIES_SECONDS}s before next city..."
            )
            time.sleep(GAP_BETWEEN_CITIES_SECONDS)

    # ── Summary ───────────────────────────────────────────────────────
    total_elapsed = round(time.time() - total_start, 1)
    total_records = sum(r["records"] for r in results)
    failed        = [r["city"] for r in results if r["status"] == "failed"]
    success       = [r["city"] for r in results if r["status"] == "success"]

    logger.info(f"\n{'='*50}")
    logger.info(f"Scraping complete in {total_elapsed}s")
    logger.info(f"  Total records: {total_records}")
    logger.info(f"  Success ({len(success)}): {success}")
    if failed:
        logger.error(f"  Failed  ({len(failed)}): {failed}")

    save_summary_report(results, total_elapsed)

    if failed:
        logger.error(
            f"{len(failed)} cities failed after {MAX_RETRIES} retries. "
            f"Check reports/scraping/ for details."
        )
        sys.exit(1)

    logger.info("Next: python src/validation/validate_schema.py --city all")


if __name__ == "__main__":
    main()