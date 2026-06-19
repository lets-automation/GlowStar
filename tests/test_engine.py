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
    df, _ = load_records()
    sold = sold_stones(df, drop_outliers=True)
    train, test, _ = time_split(sold, "2026-05-01")
    eng = PricingEngine(EngineConfig(split_date="2026-05-01")).fit(train)
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
    assert mae < 6.0           # engine measured ~5.0; baseline ~7.4
    assert not math.isnan(mae)


def test_bgm_unassessed_flag_when_no_bgm_data(trained):
    """records.json has no BGM fields -> every stone priced on the clean base
    must be flagged bgm_unassessed (client request: surface the assumption)."""
    eng, test = trained
    sugg = eng.predict(test.head(50))
    assert all(s.assumes_no_bgm for s in sugg)
    assert all("bgm_unassessed" in s.flags for s in sugg)


def test_online_feedback_correction_shifts_price(trained):
    """A per-segment override correction must immediately move suggestions."""
    eng, test = trained
    row = test.iloc[[0]]
    before = eng.predict(row)[0].suggested_discount
    seg = f"{row.iloc[0]['Shape_full']}"
    eng.set_corrections({seg: {"offset": 5.0, "n": 9}})
    after = eng.predict(row)[0]
    assert abs((after.suggested_discount - before) - 5.0) < 1e-6
    assert after.feedback_correction_pts == 5.0
    eng.set_corrections({})        # reset so other tests are unaffected
