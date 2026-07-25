"""Regression tests for the defects found in the 2026-07 audit.

Each test pins a specific bug that reached the client (or would have). They are
written to fail LOUDLY if the behaviour regresses, because every one of these was
silent: no test failed, nothing crashed, and the wrong number shipped anyway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glowstar.features.build import (BGM_FEATURES, CATEGORICAL_FEATURES,
                                     GRID_FEATURES, UNASSESSED, build_features)
from glowstar.data.loaders import parse_tinge
from glowstar.market.aggregate_bulk import _milky_severity, _shade_class
from glowstar.market.bgm import _has_bgm_fields
from glowstar.market.master_grid import MasterGrid, canon_shape
from glowstar.models.gbm import QuantileGBM
from glowstar.models.engine import EngineConfig
from glowstar.reporting.price_file import _grid_check


# --------------------------------------------------------------------------
# 1. The gate must score the config that ships.
# --------------------------------------------------------------------------
def test_serving_config_is_what_the_gate_scores():
    from glowstar.training.retrain import serving_config, _assert_gate_scores_what_ships

    class _Eng:
        cfg = serving_config("2026-06-01")

    _assert_gate_scores_what_ships(_Eng())          # must not raise

    class _Drifted:
        cfg = serving_config("2026-06-01")
    _Drifted.cfg.market_led = True                  # the exact drift that shipped
    with pytest.raises(AssertionError, match="market_led"):
        _assert_gate_scores_what_ships(_Drifted())


def test_shipped_defaults_are_the_measured_ones():
    """market_led=True shipped unmeasured (MAE 7.48 vs 3.84); the market anchor was
    monotonically harmful (lam 0.5 -> MAE 3.25 vs lam 0 -> 2.07)."""
    cfg = EngineConfig()
    assert cfg.market_led is False
    assert cfg.anchor_lambda == 0.0


# --------------------------------------------------------------------------
# 2. The discount is NOT monotone in colour/clarity — do not "fix" it.
# --------------------------------------------------------------------------
def test_discount_is_not_constrained_monotone():
    """Guards against a tempting, WRONG fix that was tried and reverted.

    Constraining discount to deepen with worse colour/clarity looks free (~40% of
    swept pairs 'inverted' -> ~1%, +0.02 MAE). But the target is a discount OFF RAP,
    and Rap already prices colour/clarity — the residual surface is genuinely
    non-monotone. In the client's OWN realized sales, 47.7% of well-supported
    adjacent colour pairs have the worse colour at a SHALLOWER discount (F/G/H are
    commercial goods and trade shallower off Rap than D/E). Constraining it enforces
    a rule their market breaks half the time.
    """
    assert QuantileGBM.MONOTONIC == {}, (
        "Do not constrain the DISCOUNT monotone in colour/clarity — Rap already "
        "prices them and the client's own sales are ~48% non-monotone. If you want "
        "this, constrain PRICE and prove it on their sales first."
    )


def test_colour_and_clarity_keep_both_encodings():
    """The ordinal carries order; the categorical carries the real per-grade
    deviations from Rap. Dropping the twin cost MAE for a non-defect."""
    assert "Color" in CATEGORICAL_FEATURES
    assert "Clarity" in CATEGORICAL_FEATURES
    x = build_features(pd.DataFrame([{
        "Shape_full": "Round", "Weight": 1.0, "Color": "G", "Clarity": "VS1",
        "CPS": "3EX", "Fluorescence": "Non", "Lab": "GIA", "Location": "IND",
        "Rap": 5000.0, "MarketSheetDate_dt": pd.Timestamp("2026-01-01"),
    }]), pd.Timestamp("2026-01-01"))
    assert {"color_ordinal", "clarity_ordinal", "Color", "Clarity"} <= set(x.columns)


def test_monotonic_cst_vector_aligns_with_columns():
    """If MONOTONIC is ever populated, the vector must be positional to x.columns —
    a misaligned vector silently constrains the WRONG feature."""
    x = pd.DataFrame({"Weight": [1.0], "color_ordinal": [2.0],
                      "clarity_ordinal": [3.0], "Rap": [5000.0]})
    g = QuantileGBM()
    assert g._monotonic_cst(x) == [0, 0, 0, 0]
    g.MONOTONIC = {"color_ordinal": -1}
    try:
        assert g._monotonic_cst(x) == [0, -1, 0, 0]
    finally:
        g.MONOTONIC = {}


# --------------------------------------------------------------------------
# 3. Tinge encoding: unassessed != clean, and never all-NaN.
# --------------------------------------------------------------------------
def test_structured_tinge_codes_parse_to_severity():
    assert parse_tinge("NO", "BR") == 0.0
    assert parse_tinge("LBR", "BR") == 1.0
    assert parse_tinge("MBR", "BR") == 2.0
    assert parse_tinge("HML", "ML") == 3.0
    assert parse_tinge("MMT", "MT") == 2.0
    assert parse_tinge("LGR", "GR") == 1.0
    assert np.isnan(parse_tinge("", "BR"))
    assert np.isnan(parse_tinge(None, "BR"))
    assert np.isnan(parse_tinge(float("nan"), "BR"))
    # a code for the WRONG attribute must not be misread as that attribute
    assert np.isnan(parse_tinge("LBR", "ML"))


def test_shade_and_green_are_model_features():
    """BgmComments carried only brown+milky; shade (tint) and green were invisible
    and are genuinely priced (shade MMT ~-9 pts on the client's own sales)."""
    for f in ("milky_ord", "brown_ord", "shade_ord", "green_ord"):
        assert f in BGM_FEATURES


def test_unassessed_tinge_is_sentinel_not_nan_and_not_clean():
    df = pd.DataFrame([{"Shape_full": "Round", "Weight": 1.0, "Color": "G",
                        "Clarity": "VS1", "CPS": "3EX", "Fluorescence": "Non",
                        "Lab": "GIA", "Location": "IND", "Rap": 5000.0,
                        "MarketSheetDate_dt": pd.Timestamp("2026-01-01")}])
    x = build_features(df, pd.Timestamp("2026-01-01"))
    for f in BGM_FEATURES:
        assert x[f].notna().all(), f"{f} must never be NaN (crashes HGB fit)"
        assert x[f].iloc[0] == UNASSESSED
        assert x[f].iloc[0] != 0.0, "unassessed must be distinct from assessed-clean"


def test_fit_survives_a_tinge_field_the_feed_stopped_sending():
    """An all-NaN numeric column hard-crashes HistGradientBoosting. If the client's
    API drops a field, the nightly retrain must not die."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame([{
        "Shape_full": "Round", "Weight": 1.0, "Color": "G", "Clarity": "VS1",
        "CPS": "3EX", "Fluorescence": "Non", "Lab": "GIA", "Location": "IND",
        "Rap": 5000.0, "FDiscount": -50 + rng.normal(0, 2),
        "MarketSheetDate_dt": pd.Timestamp("2026-01-01"),
    } for _ in range(300)])          # no tinge columns at all
    x = build_features(df, pd.Timestamp("2026-01-01"))
    QuantileGBM(coverage=0.8).fit(x, df["FDiscount"])     # must not raise


def test_missing_column_yields_a_series_not_a_scalar():
    """External stone files legitimately lack columns the training frame has."""
    df = pd.DataFrame([{"Shape_full": "Round", "Weight": 1.0, "Color": "G",
                        "Clarity": "VS1", "CPS": "3EX", "Fluorescence": "Non",
                        "Lab": "GIA", "Location": "IND", "Rap": 5000.0,
                        "MarketSheetDate_dt": pd.Timestamp("2026-01-01")}] * 3)
    x = build_features(df, pd.Timestamp("2026-01-01"), with_grid=True)
    for f in GRID_FEATURES:
        assert f in x.columns and len(x[f]) == 3


def test_nan_tinge_never_crashes_or_reads_as_assessed():
    for bad in (float("nan"), None, ""):
        assert _milky_severity(bad) == "none"
        assert _shade_class(bad) == "none"
    assert _has_bgm_fields({"Milky": float("nan")}) is False
    assert _has_bgm_fields({"Milky": "No Milky"}) is True


# --------------------------------------------------------------------------
# 4. Grid: shape canonicalisation, no collisions, no interpolated numbers.
# --------------------------------------------------------------------------
def test_grid_shape_canonicalisation_covers_both_oval_spellings():
    """The live grid spells oval BOTH 'F.OVAL' (17,190 cells) and 'OVAL' (7,245).
    A one-directional map silently misses one, and the stone then falls to the
    interpolated estimate (10.4 MAE vs 2.2 for a real cell)."""
    assert canon_shape("Oval") == canon_shape("F.OVAL") == canon_shape("OVAL")
    assert canon_shape("Sq. Emerald") == canon_shape("SQUARE EMERALD")
    assert canon_shape("Cushion") == "CUSHION"
    # junk tokens seen in the live feed must never index a cell
    for junk in ("GIA", "NONE", "", None):
        assert canon_shape(junk) is None


def test_oval_and_cushion_resolve_to_their_grid_cell():
    cells = [
        {"shape": ["F.OVAL"], "color": "G", "clarity": "VS1", "cut": "3EX",
         "fluorescence": "NON", "minWeight": 0.7, "maxWeight": 0.79,
         "discount": -52.0, "cellId": "0.7,0.79,G,VS1,3EX,NON",
         "createdDate": "2026-07-12T10:00:00"},
        {"shape": ["CUSHION"], "color": "D", "clarity": "IF", "cut": "VG",
         "fluorescence": "NON", "minWeight": 1.6, "maxWeight": 1.69,
         "discount": -65.0, "cellId": "1.6,1.69,D,IF,VG,NON",
         "createdDate": "2026-07-12T10:00:00"},
    ]
    g = MasterGrid(cells)
    assert g.lookup("Oval", 0.75, "G", "VS1", "3EX", "Non").discount == -52.0
    assert g.lookup("Cushion", 1.65, "D", "IF", "VG", "Non").discount == -65.0


def test_same_cellid_across_shapes_does_not_collide():
    """cellId carries no shape, so keying by cellId alone lets one shape's cell
    overwrite another's."""
    cid = "1.0,1.09,G,VS1,3EX,NON"
    cells = [
        {"shape": ["ROUND"], "color": "G", "clarity": "VS1", "cut": "3EX",
         "fluorescence": "NON", "minWeight": 1.0, "maxWeight": 1.09,
         "discount": -40.0, "cellId": cid, "createdDate": "2026-07-12T10:00:00"},
        {"shape": ["CUSHION"], "color": "G", "clarity": "VS1", "cut": "3EX",
         "fluorescence": "NON", "minWeight": 1.0, "maxWeight": 1.09,
         "discount": -70.0, "cellId": cid, "createdDate": "2026-07-12T10:00:00"},
    ]
    g = MasterGrid(cells)
    assert g.lookup("Round", 1.05, "G", "VS1", "3EX", "Non").discount == -40.0
    assert g.lookup("Cushion", 1.05, "G", "VS1", "3EX", "Non").discount == -70.0


def test_no_cell_shows_a_blank_grid_never_an_interpolated_guess():
    """We used to print our grid-MODEL's guess under the header "Your Master grid".
    Scored against the desk it was 10.4 MAE (52% >=5pts out) and manufactured the
    "you're 20 points off your own grid" escalation. A blank is honest."""
    class _S:
        suggested_discount = -45.0
        market_median_discount = -44.0
    out = _grid_check({}, _S(), None, predicted=-64.0)   # predicted must be IGNORED
    assert out["Your Master grid (% below Rap)"] is None
    assert out["Our vs grid (pts)"] is None
    assert "interpolat" not in out["Grid check"].lower()


# --------------------------------------------------------------------------
# 4b. Fluorescence caps are a floor on DEPTH, never a premium.
# --------------------------------------------------------------------------
def test_fluorescence_normalisation_is_idempotent():
    """Every canonical output must also be an accepted input. It wasn't: "VERY
    STRONG"/"VERY SLIGHT" were missing from FLUOR_CANON, so an already-normalised
    (or long-form) value silently became "Unknown" — losing the stone's fluorescence
    on exactly the tier where it matters most."""
    from glowstar.reference.normalize import normalize_fluorescence, FLUOR_CANON

    for canonical in set(FLUOR_CANON.values()):
        assert normalize_fluorescence(canonical) == canonical, (
            f"normalize({canonical!r}) is not idempotent")
    assert normalize_fluorescence("Vstg") == normalize_fluorescence("Very Strong")
    assert normalize_fluorescence(None) == "Unknown"


def test_strong_fluor_on_colourless_is_flagged_for_review():
    """We measurably under-discount strong fluoro on near-colourless goods (bias
    +1.47 Strong / +2.21 V.Strong vs -0.06 None) because the data is far too thin
    (~37 V.Strong stones). Flag it rather than quietly quote it too expensive."""
    from glowstar.models.engine import _is_strong_fluor, _fluor_band

    assert _is_strong_fluor("Stg") and _is_strong_fluor("Vstg")
    assert _is_strong_fluor("Strong") and _is_strong_fluor("Very Strong")
    assert not _is_strong_fluor("Non")
    assert not _is_strong_fluor("Med")      # Medium is priced, not flagged
    assert not _is_strong_fluor(None)
    # only near-colourless is at risk; fluoro is ~neutral on low colours
    assert _fluor_band("D") == "D-E" and _fluor_band("H") == "H"
    assert _fluor_band("J") == "I-M"


def test_fluor_review_note_reaches_the_client():
    from glowstar.reporting.client_report import _note

    class _S:
        flags = ["fluor_review"]
        bgm_state = "clean"
        method = "model+anchor"
    assert "fluorescence" in _note(_S()).lower()


def test_a_stone_we_ask_the_desk_to_price_is_never_High_confidence():
    """Self-contradiction check on the client-facing file: we cannot print
    "please set this discount yourself, we under-price it" next to "High
    confidence". The desk reads both columns."""
    from glowstar.reporting.client_report import _confidence

    class _S:
        method = "model+anchor"
        comparable_count = 500          # would otherwise score High
        ci_discount_low, ci_discount_high = -3.0, 3.0
        flags = ["fluor_review"]
    assert _confidence(_S()) == "Low"
    _S.flags = ["bgm_review"]
    assert _confidence(_S()) == "Low"
    _S.flags = []
    assert _confidence(_S()) == "High"   # unflagged, well-supported: still High


def test_fluor_caps_are_never_positive():
    """A cap says "do not discount fluoro DEEPER than the desk does". Derived from
    a difference of pooled medians it can come out POSITIVE on a thin cell (I-M
    Faint measured +0.5), and `maximum(pen, +0.5*1.15)` then FORCES a fluorescence
    premium — quoting a fluorescent stone shallower (pricier) than the model wanted.
    """
    from glowstar.models.engine import PricingEngine

    n = 400
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        # I-M stones where Faint fluoro happens to sell SHALLOWER than None,
        # which is what produces a positive raw cap.
        fl = "Fnt" if i % 2 else "Non"
        disc = -60.0 + (4.0 if fl == "Fnt" else 0.0) + rng.normal(0, 0.5)
        rows.append({"Color": "J", "Fluorescence": fl, "FDiscount": disc})
    caps = PricingEngine.__new__(PricingEngine)._compute_fluor_caps(pd.DataFrame(rows))
    assert caps, "expected a cap for the I-M/Faint cell"
    for key, v in caps.items():
        assert v <= 0.0, f"cap {key}={v} is positive — that forces a fluoro premium"


# --------------------------------------------------------------------------
# 5. Point-in-time grid: never leak a future edit.
# --------------------------------------------------------------------------
def test_grid_history_never_returns_a_future_edit():
    from glowstar.market.grid_history import GridHistory
    raw = {"ROUND|1.0,1.09,G,VS1,3EX,NON": [
        ["2026-05-01T00:00:00", -40.0],
        ["2026-06-01T00:00:00", -50.0],
        ["2026-07-01T00:00:00", -60.0],
    ]}
    h = GridHistory(raw)
    # before any edit -> nothing
    assert h.as_of("Round", 1.05, "G", "VS1", "3EX", "Non", "2026-04-01")[0] is None
    # strictly-before semantics: only the May edit is visible in June
    assert h.as_of("Round", 1.05, "G", "VS1", "3EX", "Non", "2026-05-15")[0] == -40.0
    assert h.as_of("Round", 1.05, "G", "VS1", "3EX", "Non", "2026-06-15")[0] == -50.0
    # the July edit must NOT leak backwards into a June sale
    assert h.as_of("Round", 1.05, "G", "VS1", "3EX", "Non", "2026-06-30")[0] == -50.0
    assert h.as_of("Round", 1.05, "G", "VS1", "3EX", "Non", "2026-07-15")[0] == -60.0


def test_grid_history_reports_cell_age():
    from glowstar.market.grid_history import GridHistory
    h = GridHistory({"ROUND|1.0,1.09,G,VS1,3EX,NON": [["2026-06-01T00:00:00", -50.0]]})
    disc, age = h.as_of("Round", 1.05, "G", "VS1", "3EX", "Non", "2026-06-21")
    assert disc == -50.0 and age == 20


def test_forward_drift_correction_is_per_family_and_shrunk():
    """The model prices at a stale level; measured out-of-time it runs ~+0.9 pts
    too SHALLOW (too expensive) — concentrated in FANCIES (+2.1) while rounds are
    unbiased (+0.01). A single global number under-corrects fancies and
    over-corrects rounds, so the correction is per shape family."""
    from glowstar.models.engine import PricingEngine

    e = PricingEngine.__new__(PricingEngine)
    fams = PricingEngine._shape_family(pd.Series(["Round", "Oval", "Pear", "Baguette"]))
    assert list(fams) == ["Round", "Fancy", "Fancy", "Other"]

    e._bias_correction = {"__global__": 0.7, "Round": 0.3, "Fancy": 2.2}
    df = pd.DataFrame({"Shape_full": ["Round", "Oval", "Baguette"]})
    assert list(e._bias_shift(df)) == [0.3, 2.2, 0.7], "unknown family must fall back to global"

    e._bias_correction = {}
    assert list(e._bias_shift(df)) == [0.0, 0.0, 0.0], "no correction must be a no-op"


def test_bias_correction_is_measured_out_of_time_not_in_train():
    """In-train bias is ~0 by construction (the model fits its own training data).
    Estimating it there produced -0.23 where the true forward bias was +2.75, and the
    correction did nothing. It must come from an inner OUT-OF-TIME slice."""
    import inspect
    from glowstar.models.engine import PricingEngine

    src = inspect.getsource(PricingEngine._fit_bias_correction)
    assert "inner_days" in src and "_fold_predict" in src
    # too little history must yield no correction rather than a noisy one
    e = PricingEngine.__new__(PricingEngine)
    e.cfg = type("C", (), {"recency_half_life": 30.0})()
    e.gbm = object()
    tiny = pd.DataFrame({"OrderDate_dt": pd.to_datetime(["2026-06-01"] * 10),
                         "Shape_full": ["Round"] * 10, "FDiscount": [-50.0] * 10})
    assert e._fit_bias_correction(tiny) == {}


def test_grid_history_banking_is_idempotent():
    """The daily job pulls an OVERLAPPING window so a missed day self-heals. That is
    only safe if re-banking the same edits is a no-op — otherwise every run inflates
    the store with duplicates and quietly corrupts the point-in-time lookup."""
    import json
    from unittest.mock import patch
    from glowstar.market.grid_history import bank_history, GridHistory

    rows = [{"cellId": "1.0,1.09,G,VS1,3EX,NON", "discount": -50.0,
             "shape": ["ROUND"], "createdDate": "2026-07-15T10:00:00"},
            {"cellId": "1.0,1.09,G,VS1,3EX,NON", "discount": -52.0,
             "shape": ["ROUND"], "createdDate": "2026-07-16T10:00:00"}]
    import tempfile, pathlib
    p = pathlib.Path(tempfile.mkdtemp()) / "h.json"
    with patch("glowstar.ingestion.diamanto.get_access_token", return_value="t"), \
         patch("glowstar.ingestion.diamanto.get_cells_history", return_value=rows):
        first = bank_history(days=2, path=p)
        again = bank_history(days=2, path=p)
    assert first["added"] == 2
    assert again["added"] == 0, "re-banking the same window must add nothing"
    store = json.loads(p.read_text())
    assert sum(len(v) for v in store.values()) == 2, "duplicate edits stored"
    h = GridHistory(store)
    assert h.as_of("Round", 1.05, "G", "VS1", "3EX", "Non", "2026-07-16")[0] == -50.0
    assert h.as_of("Round", 1.05, "G", "VS1", "3EX", "Non", "2026-07-17")[0] == -52.0


def test_attach_grid_without_history_is_a_safe_noop():
    from glowstar.market.grid_history import attach_grid
    df = pd.DataFrame([{"Shape_full": "Round", "Weight": 1.0, "Color": "G",
                        "Clarity": "VS1", "CPS": "3EX", "Fluorescence": "Non",
                        "OrderDate_dt": pd.Timestamp("2026-06-15")}])
    out = attach_grid(df, None)
    assert out["grid_discount"].isna().all()      # engine still runs, just unrouted
