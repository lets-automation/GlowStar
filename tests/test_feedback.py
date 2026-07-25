"""Tests for the human-rejection feedback store and learning."""

from __future__ import annotations

import pytest

from glowstar.feedback import store as fb
from glowstar.feedback.learning import build_corrections, correction_for, as_training_examples, reason_summary


def _rec(decision, **kw):
    base = dict(stone_id="S", decision=decision, suggested_discount=-57.0,
                suggested_net=1000.0, shape_full="Round", weight=1.0, color="G",
                clarity="SI1", rap=6500.0)
    base.update(kw)
    return fb.FeedbackRecord(**base)


def test_reject_requires_reason():
    with pytest.raises(ValueError):
        _rec("reject").validate()
    _rec("reject", reason_code="market_moved").validate()    # ok


def test_override_requires_human_price():
    with pytest.raises(ValueError):
        _rec("override", reason_code="discount_too_deep").validate()
    _rec("override", reason_code="discount_too_deep", human_discount=-51.0).validate()


def test_record_roundtrip(tmp_path):
    p = tmp_path / "decisions.jsonl"
    fb.record(_rec("accept"), path=p)
    fb.record(_rec("override", reason_code="discount_too_deep", human_discount=-51.0), path=p)
    rows = fb.load_all(p)
    assert len(rows) == 2 and rows[1]["human_discount"] == -51.0


def test_build_corrections_from_overrides():
    recs = [dict(decision="override", suggested_discount=-57.0, human_discount=-51.0,
                 shape_full="Round", weight=1.0, color="G", clarity="SI1") for _ in range(4)]
    table = build_corrections(recs, min_support=3)
    corr = correction_for(table, "Round", 1.0, "G", "SI1")
    assert corr == 6.0                       # humans priced +6 shallower -> +6 correction


def test_online_correction_does_not_fall_back_to_shape_or_global():
    recs = [dict(decision="override", suggested_discount=-57.0, human_discount=-51.0,
                 shape_full="Round", weight=1.0, color="G", clarity="SI1", cps="EX")
            for _ in range(4)]
    table = build_corrections(recs, min_support=3)
    assert correction_for(table, "Round", 0.4, "F", "VS1", "EX") == 0.0


def test_training_examples_weights_overrides_higher():
    recs = [
        dict(decision="override", suggested_discount=-57.0, human_discount=-51.0,
             shape_full="Round", weight=1.0, color="G", clarity="SI1", timestamp="2026-06-10T00:00:00+00:00"),
        dict(decision="accept", suggested_discount=-55.0,
             shape_full="Oval", weight=1.5, color="H", clarity="VS2", timestamp="2026-06-10T00:00:00+00:00"),
    ]
    X, y, w = as_training_examples(recs)
    assert len(X) == 2
    assert y[0] == -51.0 and y[1] == -55.0
    assert w[0] > w[1]                        # override up-weighted


def test_reason_summary():
    recs = [dict(decision="accept"), dict(decision="reject", reason_code="bgm_present")]
    s = reason_summary(recs)
    assert s["acceptance_rate"] == 0.5 and s["reasons"]["bgm_present"] == 1
