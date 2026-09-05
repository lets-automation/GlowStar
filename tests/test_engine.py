"""End-to-end tests for the PricingEngine.

Verifies the engine trains, produces well-formed suggestions, keeps intervals
ordered, routes rare shapes / fancy colors to fallback, and never emits a
forbidden value. Also a guardrail accuracy check so a future regression that
silently worsens the model is caught.
"""

from __future__ import annotations

import math

import pytest

from glowstar.data.loaders import load_records, sold_stones
from glowstar.models.engine import PricingEngine, EngineConfig
from glowstar.validation.backtest import time_split
from glowstar.validation import metrics as M
import numpy as np


@pytest.fixture(scope="module")
def trained():
    from glowstar.config import SETTINGS
    df, _ = load_records()
    sold = sold_stones(df, drop_outliers=True)
    split = SETTINGS.backtest_split_date
    train, test, _ = time_split(sold, split)
    eng = PricingEngine(EngineConfig(split_date=split)).fit(train)
    return eng, test


def test_suggestions_well_formed(trained):
    eng, test = trained
    sugg = eng.predict(test.head(200))
    assert len(sugg) == 200
    for s in sugg:
        assert s.ci_discount_low <= s.suggested_discount <= s.ci_discount_high
        assert s.ci_net_low <= s.suggested_net <= s.ci_net_high
        assert -95 <= s.suggested_discount <= 5
        assert s.method in {"model", "model+anchor", "fallback"}
        assert s.suggested_ppc > 0


def test_rare_and_fancy_route_to_fallback(trained):
    eng, test = trained
    # A fancy-color stone, if any in test, must be flagged and use fallback.
    fancy = test[~test["Color"].isin(list("DEFGHIJKLMN"))]
    if len(fancy):
        s = eng.predict(fancy.head(5))
        assert all("fancy_color" in x.flags and x.method == "fallback" for x in s)


def test_engine_beats_baseline_threshold(trained):
    """Guardrail: leakage-free engine MAE must stay clearly below baseline."""
    eng, test = trained
    sugg = eng.predict(test)
    pred = np.array([s.suggested_discount for s in sugg])
    mae = M.compute(pred, test).mae
    # Production-representative split (recent test window, +BGM +size-tier +competence
    # guard) measures MAE ~3.88 out-of-time (baseline ~7.2). Threshold locks in the
    # ~46% gain and catches a silent regression; 4.1 leaves a small margin.
    assert mae < 4.1
    assert not math.isnan(mae)


def test_competence_guard_defers_weak_shapes(trained):
    """A shape the model+anchor loses to the segment median on (measured out-of-time)
    must be FLAGGED for human review, and must route to the baseline WHEN THERE IS
    NOTHING BETTER TO ROUTE TO.

    A GRID CELL IS SOMETHING BETTER, and that is a change from the original rule
    ("any deferred shape must use the fallback path"). The fallback is the
    hierarchical median over shape/size/colour/clarity — it cannot see the grid at
    all — so a deferred stone carrying the desk's own current price for its exact
    cell was having that thrown away.

    Measured out-of-time on the rolling 7-day production horizon, on the 168 stones
    the guard deferred that DID have a cell:

        deferred to the median baseline   MAE 4.53
        model-priced instead              MAE 2.08     (improved on 130/168)
          Oval  5.17 -> 2.60      Pear  3.63 -> 1.35

    Round, which is never deferred, moved 1.4931 -> 1.4946 across the same runs, so
    that is not refit noise. The guard's judgement is made per SHAPE on a 60-day
    inner slice; it should not override the strongest per-STONE signal available.
    """
    eng, test = trained
    # The guard fired (data-driven, not hardcoded) and picked the known weak shape.
    assert "Sq. Emerald" in eng._defer_shapes
    # Shapes the model wins on must NOT be deferred (no over-routing to the median).
    assert "Round" not in eng._defer_shapes

    weak = test[test["Shape_full"].isin(eng._defer_shapes)]
    if not len(weak):
        return
    from glowstar.market.grid_history import attach_grid
    weak = attach_grid(weak.head(40), eng.grid_history)
    sugg = eng.predict(weak)

    # The review flag fires for EVERY deferred stone, cell or no cell — the desk is
    # always told to look. Only the routing depends on the cell.
    assert all("segment_review" in s.flags for s in sugg)

    has_cell = weak["grid_discount"].notna().tolist()
    for s, cell in zip(sugg, has_cell):
        if cell:
            assert s.method != "fallback", (
                "a deferred stone WITH a grid cell must be model-priced; the "
                "fallback cannot see the grid")
        else:
            assert s.method == "fallback", (
                "a deferred stone with NO cell has nothing better than the median")


def test_bgm_assessed_from_inventory(trained):
    """The client's live BgmComments is now in the data -> stones are ASSESSED
    (clean / bgm), not 'unassessed'. Clean (No Brown, No Milky) stones price on the
    clean base with assumes_no_bgm=False; milky/brown stones are 'bgm'; only stones
    with NO BGM data at all fall back to 'unassessed'."""
    eng, test = trained
    clean = test[(test["milky_ord"] == 0) & (test["brown_ord"] == 0)]
    if len(clean):
        s = eng.predict(clean.head(30))
        assert all(not x.assumes_no_bgm for x in s)
        assert all("bgm_unassessed" not in x.flags for x in s)
        assert all(x.bgm_state == "clean" for x in s)
    bgm = test[test["milky_ord"] > 0]
    if len(bgm):
        assert all(x.bgm_state == "bgm" for x in eng.predict(bgm.head(10)))
    unknown = test[test["milky_ord"].isna() & test["brown_ord"].isna()]
    if len(unknown):
        s = eng.predict(unknown.head(10))
        assert all(x.assumes_no_bgm for x in s)
        assert all("bgm_unassessed" in x.flags for x in s)


def test_online_feedback_correction_shifts_price(trained):
    """A SPECIFIC (>=3-level) override correction must immediately move
    suggestions. Broad shape-only/global feedback is deliberately ignored
    online (see test_broad_feedback_correction_is_ignored) so one unrepresentative
    returned batch cannot drag every stone of that shape."""
    eng, test = trained
    row = test.iloc[[0]]
    r = row.iloc[0]
    before = eng.predict(row)[0].suggested_discount
    # a supported, sufficiently-specific cell: shape|weight-decade|colour
    from glowstar.market.segments import segment_keys
    key = next(k for k in segment_keys(r["Shape_full"], r["Weight"], r["Color"],
                                       r["Clarity"], r.get("CPS")) if len(k) == 3)
    seg = "|".join(map(str, key))
    eng.set_corrections({seg: {"offset": 5.0, "n": 9}})
    after = eng.predict(row)[0]
    assert abs((after.suggested_discount - before) - 5.0) < 1e-6
    assert after.feedback_correction_pts == 5.0
    eng.set_corrections({})        # reset so other tests are unaffected


def test_broad_feedback_correction_is_ignored(trained):
    """A shape-only correction must NOT move prices: a single returned batch is
    often directionally unrepresentative, and a broad offset would drag stones in
    unrelated price cells. Measured: applying broad offsets cost +0.61 MAE."""
    eng, test = trained
    row = test.iloc[[0]]
    before = eng.predict(row)[0].suggested_discount
    eng.set_corrections({str(row.iloc[0]["Shape_full"]): {"offset": 5.0, "n": 9}})
    after = eng.predict(row)[0]
    assert after.suggested_discount == before
    assert after.feedback_correction_pts == 0.0
    eng.set_corrections({})
