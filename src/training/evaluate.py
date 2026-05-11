"""
src/training/evaluate.py
CI/CD production gate check.

Reads reports/metrics.json and exits 1 if gate fails.
Called by GitHub Actions ci.yml after training.

Run:
  python src/training/evaluate.py
"""

import json
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

METRICS_PATH      = "reports/metrics.json"
GATE_MAPE_MAX     = 22.0
GATE_R2_MIN       = 0.82


def main():
    if not Path(METRICS_PATH).exists():
        logger.error(f"metrics.json not found at {METRICS_PATH}")
        logger.error("Run: python src/training/train.py first")
        sys.exit(1)

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    cv_mape = metrics.get("cv_mape", 999.0)
    cv_r2   = metrics.get("cv_r2",   0.0)
    gate    = metrics.get("production_gate", "FAIL")

    logger.info(f"CV MAPE:         {cv_mape:.4f}%  (max allowed: {GATE_MAPE_MAX}%)")
    logger.info(f"CV R²:           {cv_r2:.4f}  (min required: {GATE_R2_MIN})")
    logger.info(f"Production gate: {gate}")

    if cv_mape > GATE_MAPE_MAX or cv_r2 < GATE_R2_MIN or gate != "PASS":
        logger.error("Gate FAILED. Stopping CI/CD pipeline.")
        sys.exit(1)

    logger.info("Gate PASSED. Proceeding to Docker build and deploy.")


if __name__ == "__main__":
    main()