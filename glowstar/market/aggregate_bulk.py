"""Stream the 6.2GB Uni bulk dump into compact market-reference artifacts.

Produces, from real market listings (asking prices):
  1. artifacts/market_segments.json — per-segment median market discount for
     CLEAN (no-BGM) stones at 4 granularities, with counts. This is the
     market-level ANCHOR the pricing model uses to re-center its discount to
     today's market (the falling-/shifting-market correction, brief Sec 5/14).
  2. artifacts/bgm_discounts.json — how much extra discount BGM, each milky
     severity, and each shade class carry vs clean stones (the soft-attribute
     value learned from market data, brief Sec 5).

Cleaning applied (brief Sec, market-data authenticity):
  * Dedupe by certificate number (virtual/double-listed stones inflate market).
  * Drop rows without a usable discount or core 4C.
  * Median (not mean) aggregation; thin segments fall back to coarser levels.

Memory is bounded by storing integer-binned discount histograms per segment,
not raw values. Run:  python -m glowstar.market.aggregate_bulk
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import ijson

from ..config import ARTIFACTS_DIR, PATHS
from .segments import segment_keys

log = logging.getLogger(__name__)

# Discount histogram domain (percent, clamped). 1-point bins are plenty for a
# median anchor and keep memory tiny.
_DISC_MIN, _DISC_MAX = -100, 20


def _clamp_bin(d: float) -> int:
    return min(_DISC_MAX, max(_DISC_MIN, int(round(d))))


def _median_from_hist(hist: dict[int, int]) -> tuple[float, int]:
    """Median and total count from an integer histogram."""
    n = sum(hist.values())
    if n == 0:
        return float("nan"), 0
    target = (n + 1) / 2.0
    cum = 0
    for k in sorted(hist):
        cum += hist[k]
        if cum >= target:
            return float(k), n
    return float("nan"), n


def _clean_text(v) -> str:
    """Lower-cased text from any source value; '' for absent.

    Tolerates float NaN, which is what a MISSING cell reads as once these fields
    come from a DataFrame rather than raw JSON. `(v or "")` does NOT handle it —
    NaN is truthy, so it slips through and blows up on .strip().
    """
    if v is None:
        return ""
    if isinstance(v, float) and v != v:          # NaN
        return ""
    return str(v).strip().lower()


def _shade_class(shade) -> str:
    """Bucket the raw shade into value-negative / neutral / positive.

    Brown/Green/Gray (and faint variants) and Mix are value-negative true BGM
    shades. Yellow is the normal graded-color axis (neutral here). Pink and
    other fancy hues are value-positive. None/Not Reported -> none.
    """
    s = _clean_text(shade)
    if not s or s in ("none", "not reported", "-"):
        return "none"
    if any(t in s for t in ("brown", "green", "gray", "grey", "mix")):
        return "negative"
    if any(t in s for t in ("pink", "blue", "orange", "purple", "red", "violet")):
        return "positive"
    return "neutral"  # e.g. yellow / unclassified


def _milky_severity(milky) -> str:
    m = _clean_text(milky)
    if not m or m in ("no milky", "not reported", "none"):
        return "none"
    if "heavy" in m:
        return "heavy"
    if "medium" in m:
        return "medium"
    if "slight" in m or "light" in m:
        return "slight"
    return "none"


def aggregate(path: Path | None = None, limit: int | None = None) -> dict:
    path = Path(path) if path else PATHS.uni_bulk
    seen_certs: set[str] = set()

    # Per-segment histograms for CLEAN stones (the anchor).
    clean_hist: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    # Global histograms by soft-attribute level (the deltas).
    by_bgm: dict[str, dict[int, int]] = {"Yes": defaultdict(int), "No": defaultdict(int)}
    by_milky: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    by_shade: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    n_seen = n_used = n_dup = n_nodisc = 0
    truncated = False

    with open(path, "rb") as fh:
        stream = ijson.items(fh, "item", use_float=True)
        while True:
            try:
                item = next(stream)
            except StopIteration:
                break
            except ijson.common.IncompleteJSONError:
                # The bulk dump is truncated at the tail (premature EOF). Use
                # every complete record parsed so far rather than discarding the
                # whole run; record that it was truncated.
                truncated = True
                log.warning("Bulk file truncated after %s records — using partial data.",
                            f"{n_seen:,}")
                break
            n_seen += 1
            if limit and n_seen > limit:
                break

            cert = (item.get("certificateNumber") or "").strip()
            uid = cert or str(item.get("diamondID") or "")
            if uid:
                if uid in seen_certs:
                    n_dup += 1
                    continue
                seen_certs.add(uid)

            price = item.get("price") or {}
            disc = price.get("listDiscount")
            shape = item.get("shape")
            size = item.get("size")
            color = item.get("color")
            clarity = item.get("clarity")
            if disc is None or shape is None or size is None or color is None or clarity is None:
                n_nodisc += 1
                continue
            try:
                b = _clamp_bin(float(disc))
                w = float(size)
            except (TypeError, ValueError):
                n_nodisc += 1
                continue

            n_used += 1
            is_bgm = (item.get("is_bgm") or "No").strip()
            milky = _milky_severity(item.get("milky"))
            shade = _shade_class(item.get("Shade"))

            by_bgm["Yes" if is_bgm == "Yes" else "No"][b] += 1
            by_milky[milky][b] += 1
            by_shade[shade][b] += 1

            # Anchor uses CLEAN stones only (no BGM, no milky, neutral/none shade).
            # Pass cut so the banked market is CUT-AWARE: a VG stone matches the VG
            # market (which trades ~10 pts deeper) instead of the EX-dominated blend.
            if is_bgm != "Yes" and milky == "none" and shade in ("none", "neutral"):
                cut = item.get("cut") or item.get("cutShortTitle")
                for key in segment_keys(shape, w, color, clarity, cut):
                    clean_hist[key][b] += 1

            if n_seen % 250_000 == 0:
                log.info("…%s seen, %s used, %s dup, %s skipped",
                         f"{n_seen:,}", f"{n_used:,}", f"{n_dup:,}", f"{n_nodisc:,}")

    # Materialize medians.
    segments = {}
    for key, hist in clean_hist.items():
        med, cnt = _median_from_hist(hist)
        segments["|".join(map(str, key)) if key else "__global__"] = {"median_discount": med, "n": cnt}

    def deltas(table):
        out = {}
        base_med, _ = _median_from_hist(by_bgm["No"])  # clean BGM=No as reference
        for lvl, hist in table.items():
            med, cnt = _median_from_hist(hist)
            out[lvl] = {"median_discount": med, "n": cnt,
                        "delta_vs_clean": round(med - base_med, 2)}
        return out

    bgm_report = {
        "reference_clean_median": _median_from_hist(by_bgm["No"])[0],
        "by_bgm": deltas(by_bgm),
        "by_milky": deltas(by_milky),
        "by_shade": deltas(by_shade),
        "counts": {"seen": n_seen, "used": n_used, "duplicates": n_dup,
                   "skipped": n_nodisc, "truncated_source": truncated},
    }
    return {"segments": segments, "bgm": bgm_report}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = aggregate()
    (ARTIFACTS_DIR / "market_segments.json").write_text(
        json.dumps(out["segments"], indent=0), encoding="utf-8")
    (ARTIFACTS_DIR / "bgm_discounts.json").write_text(
        json.dumps(out["bgm"], indent=2), encoding="utf-8")
    c = out["bgm"]["counts"]
    log.info("DONE: %s segments | used %s of %s listings (%s dup, %s skipped)",
             f"{len(out['segments']):,}", f"{c['used']:,}", f"{c['seen']:,}",
             f"{c['duplicates']:,}", f"{c['skipped']:,}")
    log.info("BGM reference clean median discount: %.1f%%", out["bgm"]["reference_clean_median"])
    for lvl, v in out["bgm"]["by_milky"].items():
        log.info("  milky=%s: median %.1f%% (delta %+.1f, n=%s)",
                 lvl, v["median_discount"], v["delta_vs_clean"], f"{v['n']:,}")
    for lvl, v in out["bgm"]["by_shade"].items():
        log.info("  shade=%s: median %.1f%% (delta %+.1f, n=%s)",
                 lvl, v["median_discount"], v["delta_vs_clean"], f"{v['n']:,}")


if __name__ == "__main__":
    main()
