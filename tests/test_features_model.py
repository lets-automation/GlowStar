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
