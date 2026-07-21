"""Central configuration: data-file locations and runtime settings.

No secrets live here. API credentials are read from environment variables
(see .env.example) and never hardcoded (brief Section 11).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repository root = parent of the `glowstar` package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the REPO ROOT explicitly (not the process cwd). A bare
# load_dotenv() searches upward from the current working directory, so a
# scheduler-invoked job (Windows Task / cron) running from C:\Windows\System32
# or /root would NOT find the repo's .env — credentials would silently read as
# empty and every live pull would abort. Pinning the path makes the snapshot
# job authenticate correctly regardless of the working directory it is launched
# from. Any real environment variables already set still take precedence.
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:  # python-dotenv optional at import time
    pass

# Where immutable snapshots and derived artifacts are written.
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def _p(name: str, default: Path) -> Path:
    """Resolve a data-file path, overridable by environment variable."""
    return Path(os.environ.get(name, str(default)))


@dataclass(frozen=True)
class Paths:
    """Locations of the reference data files shipped with the brief."""

    records: Path = field(default_factory=lambda: _p("GS_RECORDS", REPO_ROOT / "records.json"))
    rap_round: Path = field(default_factory=lambda: _p("GS_RAP_ROUND", REPO_ROOT / "CSV2_ROUND_8_4.csv"))
    rap_pear: Path = field(default_factory=lambda: _p("GS_RAP_PEAR", REPO_ROOT / "CSV2_PEAR_8_4.csv"))
    rap_history: Path = field(default_factory=lambda: _p("GS_RAP_HISTORY", REPO_ROOT / "Rap_history (1).csv"))
    uni_sample: Path = field(default_factory=lambda: _p("GS_UNI_SAMPLE", REPO_ROOT / "response-uni.json"))
    uni_bulk: Path = field(default_factory=lambda: _p("GS_UNI_BULK", REPO_ROOT / "UNI_BGMFILEDWITH.json"))
    cells_history: Path = field(default_factory=lambda: _p("GS_CELLS_HISTORY", REPO_ROOT / "response.json"))


PATHS = Paths()


@dataclass(frozen=True)
class Settings:
    """Tunable thresholds. Defaults are conservative; all are explainable."""

    # Out-of-time validation split. Train on sales strictly before this date;
    # test on sales on/after it. Kept RECENT so the test window mirrors production
    # (nightly retrain -> predict the near term); as live data grows, move this
    # forward. ISO date string.
    backtest_split_date: str = os.environ.get("GS_BACKTEST_SPLIT", "2026-06-01")

    # A segment with fewer than this many training sales routes to fallback
    # (brief Section 7.6).
    min_segment_samples: int = int(os.environ.get("GS_MIN_SEGMENT_SAMPLES", "30"))

    # Confidence-interval coverage target for the quantile model (e.g. 0.80
    # => predict the 10th and 90th percentiles).
    interval_coverage: float = float(os.environ.get("GS_INTERVAL_COVERAGE", "0.80"))

    # Stones at/above this net value are always flagged for human review
    # regardless of model confidence (brief Section 2.4).
    high_value_usd: float = float(os.environ.get("GS_HIGH_VALUE_USD", "50000"))

    # Outlier guardrails for FDiscount (brief Section 4.3): valid final
    # discounts are negative and bounded; anything outside is winsorized/logged.
    fdiscount_min: float = float(os.environ.get("GS_FDISCOUNT_MIN", "-90"))
    fdiscount_max: float = float(os.environ.get("GS_FDISCOUNT_MAX", "0"))


SETTINGS = Settings()
