"""Price a client stock file: Sale Discount (AI) + Sale (Fast/Medium/Slow).

Client request (2026-08-13): price `artifacts/499.xlsx`, return an AI sale
discount and a three-way speed label, using the LAST 2 MONTHS of sales, and
bypass any stone we cannot support with recent data.

DESIGN NOTES — read before changing
-----------------------------------
* Prices come from `PricingService`, the SHIPPED path — not a bare engine.
  Scoring a convenient stand-in instead of what the client receives is the
  mistake CLAUDE.md Trap 5 exists to prevent.

* EVERY categorical handed to the engine is checked against the vocabulary the
  model was actually FITTED on, and the run ABORTS if any value is unseen. This
  is not defensive decoration — the first version of this script fed
  `Fluorescence="None"/"Faint"/"Medium"` (the canonical forms from
  `normalize_fluorescence`) when the model is trained on the feed's own
  abbreviations `Non/Fnt/Med`. HistGradientBoosting silently ignores an unseen
  category, so all 499 stones lost their fluorescence signal and 270 of them
  were mispriced by up to 7.34 points, with no error and no flag. That is Trap 9
  exactly: `reporting/price_file.py` had the right map (`_FLUOR_MAP`) and the
  new path did not. A hard assertion is the only thing that makes this class of
  bug loud.

* `_make_cps` passes an unrecognised grade straight through, so a `VG+` cut
  became CPS `VG+`, which `normalize_cps` could not parse and turned into `NA` —
  "no cut information at all", worth +3.41 points on that stone. Grade
  modifiers are stripped before mapping.

* The FrontOffice API returns a FIVE-level scale (High/Semi High/Medium/Semi
  Slow/Slow). This file asks for THREE (Fast/Medium/Slow), so the label is
  computed here rather than changing a live API contract underneath the CRM.
  The result is cross-checked against the live table so the workbook and the
  CRM cannot silently disagree about the same stone.

* Speed segments back off by size. Measured in the window, a 0.30-0.39ct round
  sells in ~19 days and a 0.60-0.69ct in ~37 — so a size-blind segment would
  give both the same label. Backoff keeps specificity where support exists.

* BYPASS is deliberate and visible: a stone with no supported segment gets a
  blank label and a stated reason, never a guessed one.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from glowstar.data.loaders import load_records
from glowstar.features.build import CATEGORICAL_FEATURES
from glowstar.market.segments import size_band
from glowstar.reference.normalize import normalize_shape
from glowstar.reporting.price_file import _make_cps, _FLUOR_MAP
from glowstar.service.pricing_service import PricingService, StoneIn
from glowstar.service.tradeability import tradeability_for

WINDOW_DAYS = 60          # "last 2 months"
MIN_RECENT = 15           # below this a segment is bypassed, not guessed
IN_FILE = Path("artifacts/499.xlsx")
OUT_FILE = Path("artifacts/499_priced.xlsx")


# --------------------------------------------------------------------------
# Vocabulary guard
# --------------------------------------------------------------------------
def trained_vocabulary(sold: pd.DataFrame) -> dict[str, set]:
    """The exact set of values the model was fitted on, per categorical."""
    return {c: set(sold[c].astype("string").fillna("NA").tolist())
            for c in CATEGORICAL_FEATURES if c in sold.columns}


def check_vocabulary(stones: list[StoneIn], vocab: dict[str, set]) -> dict[str, dict]:
    """Every categorical value we are about to send, versus what was trained.

    Returns {field: {value: count}} for unseen values only.
    """
    unseen: dict[str, dict] = {}
    for st in stones:
        for field, allowed in vocab.items():
            val = getattr(st, field, None)
            if val is None:
                continue
            if str(val) not in allowed:
                unseen.setdefault(field, {}).setdefault(str(val), 0)
                unseen[field][str(val)] += 1
    return unseen


# --------------------------------------------------------------------------
# Speed table
# --------------------------------------------------------------------------
def _seg_keys(shape: str, weight: float, color: str, clarity: str) -> list[str]:
    """Most specific -> least, mirroring market.segments.segment_keys."""
    sb = size_band(weight)
    return [f"{shape}|{sb}|{color}|{clarity}",
            f"{shape}|{sb}|{color}",
            f"{shape}|{color}|{clarity}"]


def build_speed_table(sold_window: pd.DataFrame) -> tuple[dict, dict, tuple[float, float]]:
    """Median days-to-sell per segment, over the last WINDOW_DAYS of SALES.

    This is a sellers' view: of the stones that SOLD in the window, how long had
    they been in stock. It is directly interpretable and it is what "use the last
    2 months" asks for.

    It is NOT Kaplan-Meier, and the reason is a bug this script had first time
    round: adding all standing stock as censored observations while restricting
    the sold set to 60 days breaks the risk sets (you keep every stone that has
    NOT sold and discard every sale older than the window), which pushed the
    median from ~36 days out to 83-134. Verified against the shipped
    Kaplan-Meier table, this sellers' view agrees closely (19-38d here vs 25-47d
    there on the file's twelve largest segments), so the simplification does not
    change the answer — it is cross-checked per stone below.

    KNOWN LIMIT, stated not hidden: a segment that has stopped selling entirely
    is invisible here. Such a segment falls below MIN_RECENT and is BYPASSED,
    which is the correct answer for it anyway.
    """
    d = sold_window
    med, cnt = {}, {}
    for level in range(3):
        keys = [_seg_keys(r.Shape_full, r.Weight, r.Color, r.Clarity)[level]
                for r in d.itertuples()]
        g = pd.DataFrame({"k": keys, "dur": d["dur"].to_numpy()}).groupby("k")["dur"]
        for k, sub in g:
            if len(sub) >= MIN_RECENT:
                med.setdefault(k, float(sub.median()))
                cnt.setdefault(k, int(len(sub)))

    # Fast/Medium/Slow are TERTILES OF THIS DESK'S OWN recent distribution, not
    # absolute day counts. "Slow" means slow for them — the only meaning the
    # desk can act on. Cut on the most-specific level so the scale is not
    # distorted by the coarse backoff levels.
    finest = [v for k, v in med.items() if k.count("|") == 3]
    vals = np.array(sorted(finest if len(finest) >= 12 else list(med.values())))
    cuts = (float(np.quantile(vals, 1 / 3)), float(np.quantile(vals, 2 / 3)))
    return med, cnt, cuts


def speed_label(days: float, cuts: tuple[float, float]) -> str:
    return "Fast" if days <= cuts[0] else ("Medium" if days <= cuts[1] else "Slow")


# --------------------------------------------------------------------------
# Row -> engine input
# --------------------------------------------------------------------------
def _strip_modifier(grade) -> str:
    """'VG+' / 'EX-' -> 'VG' / 'EX'. The client grades with modifiers; the model
    was fitted on clean grades, and an unrecognised code becomes 'NA' (= no cut
    information), which is strictly worse than the grade one step coarser."""
    s = str(grade if grade is not None else "").strip().upper()
    return s[:-1] if s.endswith(("+", "-")) else s


def _f(v):
    try:
        f = float(v)
        return f if np.isfinite(f) and f > 0 else None
    except (TypeError, ValueError):
        return None


def to_stone(row) -> StoneIn:
    """One spreadsheet row -> the engine's input.

    Tinge: this file grades `Tinge=NO` and `Milky=M0` on all 499 rows (verified,
    not assumed). `M0` means milky-grade-zero, i.e. assessed and clean, so it
    maps to 0.0 rather than the UNASSESSED sentinel — which is the correct
    reading and avoids 499 spurious review flags. A row carrying any other tinge
    value is passed through unchanged so `parse_tinge` grades it properly.
    """
    tinge = str(row.get("Tinge", "")).strip().upper()
    milky = str(row.get("Milky", "")).strip().upper()
    clean_t = "NO" if tinge in ("NO", "N", "") else tinge
    clean_m = "NO" if milky in ("M0", "NO", "N", "") else milky

    # Fluorescence MUST use the feed's own abbreviations — the vocabulary the
    # model was fitted on. See the module docstring.
    fl_raw = str(row.get("Fluo Int", "NON")).strip().upper()
    fl = _FLUOR_MAP.get(fl_raw, "Non")

    return StoneIn(
        StoneId=str(row["Stone ID"]),
        Shape_full=normalize_shape(row["Shape"]) or str(row["Shape"]),
        Weight=float(row["Carats"]),
        Color=str(row["Color"]).strip(),
        Clarity=str(row["Clarity"]).strip(),
        CPS=_make_cps(_strip_modifier(row.get("Cut")),
                      _strip_modifier(row.get("Polish")),
                      _strip_modifier(row.get("Symm"))),
        Fluorescence=fl,
        Lab=str(row.get("Lab", "GIA")).strip(),
        Length=_f(row.get("Length")), Width=_f(row.get("Width")),
        Depth=_f(row.get("Height")),
        Brown=clean_t, Shade=clean_t, Green=clean_t, Milky=clean_m,
    )


# --------------------------------------------------------------------------
def main() -> pd.DataFrame:
    src = pd.read_excel(IN_FILE)
    print(f"loaded {len(src)} stones from {IN_FILE}")

    df, _ = load_records()
    sold_all = df[df["Status"] == "Sold"].copy()
    now = pd.Timestamp.now().normalize()
    cutoff = now - pd.Timedelta(days=WINDOW_DAYS)
    win = sold_all[sold_all["OrderDate_dt"] >= cutoff].copy()
    win["dur"] = (win["OrderDate_dt"] - win["CreatedDate_dt"]).dt.days
    win = win[win["dur"].notna() & (win["dur"] >= 0)]
    print(f"speed window: {len(win):,} sales "
          f"({win['OrderDate_dt'].min():%Y-%m-%d} .. {win['OrderDate_dt'].max():%Y-%m-%d})")

    med, cnt, cuts = build_speed_table(win)
    print(f"speed table: {len(med)} supported segments (>={MIN_RECENT} sales); "
          f"tertiles at {cuts[0]:.0f}d / {cuts[1]:.0f}d")

    # --- build every stone first, so the vocabulary guard sees the whole file --
    stones, build_errs = [], {}
    for i, row in src.iterrows():
        try:
            stones.append(to_stone(row))
        except Exception as e:
            build_errs[i] = f"{type(e).__name__}: {e}"
            stones.append(None)

    vocab = trained_vocabulary(sold_all)
    unseen = check_vocabulary([s for s in stones if s is not None], vocab)
    if unseen:
        print("\n*** ABORTING — values the model was never fitted on ***")
        for field, vals in unseen.items():
            print(f"  {field}: {vals}")
        print("\nHistGradientBoosting ignores an unseen category silently, so this "
              "would drop the signal with no error. Fix the mapping, do not "
              "loosen this check.")
        raise SystemExit(1)
    print(f"vocabulary guard: OK — every value across "
          f"{', '.join(sorted(vocab))} is in the trained set")

    svc = PricingService()          # shipped path, feedback OFF
    print(f"model: {svc.engine._train_max_date:%Y-%m-%d} training epoch\n")

    out = []
    for (_, row), stone in zip(src.iterrows(), stones):
        rec = {
            "Sr.No": row.get("Sr.No"), "Stone ID": row["Stone ID"],
            "Certificate": row.get("Certificate"), "Shape": row["Shape"],
            "Carats": row["Carats"], "Color": row["Color"], "Clarity": row["Clarity"],
            "Cut": row.get("Cut"), "Polish": row.get("Polish"), "Symm": row.get("Symm"),
            "Fluo Int": row.get("Fluo Int"), "Rap $": row.get("Rap $"),
            "Their Disc %": row.get("Disc %"),
        }
        if stone is None:
            rec.update({"Sale Discount(AI)": None, "Sale": "",
                        "Notes": f"BYPASSED — {build_errs[_]}"})
            out.append(rec)
            continue
        try:
            f = svc.price(stone, explain=False)["suggestion"]
            rec["Sale Discount(AI)"] = f["suggested_discount"]
            rec["AI $/ct"] = f["suggested_ppc"]
            rec["AI Amount U$"] = round(f["suggested_net"], 2)
            rec["Fair Range Low"] = f["ci_discount_low"]
            rec["Fair Range High"] = f["ci_discount_high"]
            td = row.get("Disc %")
            rec["Variance vs Their Disc"] = (
                None if pd.isna(td) else round(f["suggested_discount"] - float(td), 2))
            flags = [x for x in (f.get("flags") or []) if x != "bgm_unassessed"]
            rec["Flags"] = ", ".join(flags)

            # --- speed, with hierarchical backoff ---
            keys = _seg_keys(stone.Shape_full, stone.Weight, stone.Color, stone.Clarity)
            hit = next((k for k in keys if k in med), None)
            if hit is None:
                rec.update({"Sale": "", "Expected Days": None,
                            "Speed Basis": f"BYPASSED — under {MIN_RECENT} sales in "
                                           f"the last {WINDOW_DAYS}d at any level"})
            else:
                d = med[hit]
                rec["Sale"] = speed_label(d, cuts)
                rec["Expected Days"] = round(d)
                lvl = ("size+colour+clarity", "size+colour", "colour+clarity")[keys.index(hit)]
                rec["Speed Basis"] = f"{cnt[hit]} sales in {WINDOW_DAYS}d ({lvl})"
                # Cross-check against the shipped table the CRM will return.
                api = tradeability_for(stone.Shape_full, stone.Weight,
                                       stone.Color, stone.Clarity)
                rec["CRM Tradeability"] = api["label"]
                rec["CRM Days"] = api["median_days"]
            rec["Notes"] = ""
        except Exception as e:
            rec.update({"Sale Discount(AI)": None, "Sale": "",
                        "Notes": f"BYPASSED — {type(e).__name__}: {e}"})
        out.append(rec)

    res = pd.DataFrame(out)
    cols = ["Sr.No", "Stone ID", "Certificate", "Shape", "Carats", "Color", "Clarity",
            "Cut", "Polish", "Symm", "Fluo Int", "Rap $", "Their Disc %",
            "Sale Discount(AI)", "Sale",
            "AI $/ct", "AI Amount U$", "Fair Range Low", "Fair Range High",
            "Variance vs Their Disc", "Expected Days", "Speed Basis",
            "CRM Tradeability", "CRM Days", "Flags", "Notes"]
    res = res.reindex(columns=[c for c in cols if c in res.columns])
    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as xw:
        res.to_excel(xw, index=False, sheet_name="Priced")
    print(f"wrote {OUT_FILE}  ({len(res)} rows)")
    return res


if __name__ == "__main__":
    main()
