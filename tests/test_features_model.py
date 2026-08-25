"""Tests for the leakage-free feature pipeline and the quantile model.

The leakage guard is the single most important correctness property of the
pricing engine (brief Section 7.4): if a transaction-derived column reaches the
model, the accuracy claim is invalid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glowstar.data.loaders import FORBIDDEN_FEATURES, load_records, sold_stones
from glowstar.features.build import build_features, get_target, _assert_no_leakage
from glowstar.models.baseline import HierarchicalMedianModel
from glowstar.models.gbm import QuantileGBM


@pytest.fixture(scope="module")
def sold():
    df, _ = load_records()
    return sold_stones(df, drop_outliers=True)


def test_feature_matrix_has_no_forbidden_columns(sold):
    x = build_features(sold)
    assert FORBIDDEN_FEATURES.isdisjoint(x.columns)
    # AvailableDays / Ageing are intentionally excluded (post-hoc leakage).
    assert "AvailableDays" not in x.columns
    assert "Ageing" not in x.columns


def test_leakage_guard_trips_on_forbidden_column():
    bad = pd.DataFrame({"Weight": [1.0], "FDiscount": [-50.0]})
    with pytest.raises(AssertionError, match="Leakage guard"):
        _assert_no_leakage(bad)


def test_target_is_fdiscount(sold):
    """`get_target` returns FDiscount as float — including its gaps.

    This used to assert `(y == sold["FDiscount"]).all()`, which quietly assumed
    the target is never missing. That stopped being true: the client's feed
    returns a small number of SOLD stones with no FDiscount (1 row in the
    server's training split on 2026-08-11, 3 locally). Because `NaN != NaN`, the
    old form fails the moment a single one appears — it was asserting a property
    of the DATA, not of the function.

    The missing values are real and are filtered at fit time (see
    `PricingEngine.fit`), not here. So compare where values exist, and assert the
    gaps line up exactly.
    """
    y = get_target(sold)
    assert y.isna().equals(sold["FDiscount"].isna()), "get_target changed which values are missing"
    present = sold["FDiscount"].notna()
    assert (y[present] == sold["FDiscount"][present]).all()


def test_baseline_predicts_and_reports_basis(sold):
    train = sold.iloc[:8000]
    model = HierarchicalMedianModel().fit(train)
    preds = model.predict_detailed(sold.iloc[8000:8050])
    assert len(preds) == 50
    for p in preds:
        assert -90 <= p.discount <= 0
        assert p.count >= 0


def test_quantile_model_orders_intervals(sold):
    train = sold.iloc[:6000]
    x, y = build_features(train), get_target(train)
    gbm = QuantileGBM(coverage=0.8).fit(x, y)
    lo, mid, hi = gbm.predict_interval(build_features(sold.iloc[6000:6200]))
    assert np.all(lo <= mid) and np.all(mid <= hi)   # well-ordered after sort


# --- a pickled model must be served with the schema it was TRAINED with -----
# REAL BUG, caught pre-deploy (2026-08-25). Adding the shape-free backoff levels
# to `_LEVELS` broke every model ALREADY in the registry: `_predict_row` walked
# the module-level constant (now 8 entries) while `self._tables`, built at fit
# time, still had 5. Result: IndexError -> HTTP 500 on every stone routed to the
# baseline fallback, from the moment the code deployed until the next retrain.
# Nothing in the suite caught it because synthetic stones never reach the
# fallback; it took pricing REAL stones through the shipped endpoint.
#
# THE ROW MUST MATCH NOTHING. A first version of this test passed with the bug
# reintroduced, because its rows matched at level 0 and the walk never reached
# the out-of-range index. These tests predict a stone that misses EVERY level,
# which is the only path that actually indexes past the end of `_tables`.

def _toy_frame():
    import pandas as pd
    n = 60
    return pd.DataFrame({
        "Shape_full": ["Round"] * n,
        "Weight": [0.5 + 0.01 * i for i in range(n)],
        "Color": ["G"] * n,
        "Clarity": ["VS1"] * n,
        "FDiscount": [-40.0 - (i % 5) for i in range(n)],
    })


def _unmatchable_frame():
    """Nothing in `_toy_frame` shares any key with this, at any level."""
    import pandas as pd
    return pd.DataFrame({
        "Shape_full": ["Hexagonal"],
        "Weight": [7.77],
        "Color": ["ZZ"],
        "Clarity": ["QQ"],
    })


def test_old_pickle_survives_new_levels_being_added():
    """Simulate the exact production shape: fit, THEN the code grows a level."""
    from glowstar.models import baseline as B

    model = B.HierarchicalMedianModel(min_samples=5).fit(_toy_frame())
    n_before = len(model._tables)

    # Insert BEFORE the trailing `()` global sentinel — that is where the F3
    # change actually added levels, and `_predict_row` breaks at the sentinel,
    # so anything appended after it is unreachable and the test proves nothing.
    orig = B._LEVELS
    assert orig[-1] == (), "the global sentinel must stay last for this to be realistic"
    B._LEVELS = orig[:-1] + (("Color", "Clarity"), ("Color",)) + orig[-1:]
    try:
        preds = model.predict(_unmatchable_frame())   # must NOT raise IndexError
    finally:
        B._LEVELS = orig

    assert len(preds) == 1
    assert preds[0] == preds[0], "no NaN prediction"
    assert len(model._tables) == n_before, "predicting must not mutate the model"


def test_pre_levels_pickle_without_the_attribute_still_predicts():
    """A pickle from before `_levels` existed has no such attribute at all."""
    from glowstar.models import baseline as B

    model = B.HierarchicalMedianModel(min_samples=5).fit(_toy_frame())
    del model._levels                                  # exactly what unpickling gives
    model._tables = model._tables[:4]                  # ...and a shorter table list
    preds = model.predict(_unmatchable_frame())
    assert len(preds) == 1 and preds[0] == preds[0]


def test_a_freshly_fit_model_uses_every_current_level():
    """The compatibility shim must not silently disable the F3 fix on new models."""
    from glowstar.models import baseline as B
    model = B.HierarchicalMedianModel(min_samples=5).fit(_toy_frame())
    assert model._levels == B._LEVELS
    assert len(model._tables) == len(B._LEVELS)
    assert model._fitted_levels() == B._LEVELS
