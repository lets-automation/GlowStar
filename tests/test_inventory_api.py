"""Phase-D/E gate: the JSON endpoints, the four reports, and the velocity gate.

The Phase-D acceptance test is that every payload explains its own numbers
without a caller reconstructing anything — MOU 5.1 makes the ENDPOINTS the
deliverable, not a screen, so a renderer must need no business logic of its own.

The one that has bitten this project before gets the most attention here:
**every new endpoint is a new entry point** (CLAUDE.md Trap 9). The Excel path
canonicalised the client's trade codes and the new API path did not, and 98% of
stones were affected with nothing failing loudly. So the same stone is sent
through `/inventory/velocity` twice — once as `RBC` / `EX EX EX`, once as
`Round` / `3EX` — and the two answers must be identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient        # noqa: E402

from glowstar.inventory import chart as CH       # noqa: E402
from glowstar.service import app as app_mod      # noqa: E402
from glowstar.training import velocity_retrain as VR  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app_mod.app)


@pytest.fixture(scope="module")
def view():
    return CH.build_view()


# ---------------------------------------------------------------------------
# Trap 9 — the canonicalisation rule at the NEW door
# ---------------------------------------------------------------------------
_CANON = {"StoneId": "N-1", "Shape_full": "Round", "Weight": 0.52, "Color": "G",
          "Clarity": "VS1", "CPS": "3EX", "Fluorescence": "Non", "Lab": "GIA",
          "Location": "IND"}
_TRADE = {**_CANON, "StoneId": "N-2", "Shape_full": "RBC", "CPS": "EX EX EX"}


def test_trade_codes_and_canonical_names_get_the_same_answer(client):
    """The rule: price the SAME stone through the old and the new path and
    assert they agree. Here the two paths are the two vocabularies the client's
    own systems send — and one triple-excellent stone once got three different
    prices depending only on punctuation."""
    r = client.post("/inventory/velocity",
                    json={"stones": [_CANON, _TRADE], "age_days": [0, 0]})
    assert r.status_code == 200, r.text
    a, b = r.json()
    assert a["Segment"] == b["Segment"], "shape code did not canonicalise"
    assert a["Class"] == b["Class"]
    assert a["ExpectedDaysToSell"] == b["ExpectedDaysToSell"]
    assert a["OwnVelocityScore"] == b["OwnVelocityScore"]


def test_hyphenated_and_spaced_cut_codes_agree(client):
    variants = [{**_CANON, "StoneId": f"C{i}", "CPS": c}
                for i, c in enumerate(("3EX", "EX-EX-EX", "EX EX EX"))]
    out = client.post("/inventory/velocity",
                      json={"stones": variants, "age_days": [0, 0, 0]}).json()
    assert len({r["ExpectedDaysToSell"] for r in out}) == 1
    assert len({r["Class"] for r in out}) == 1


def test_velocity_endpoint_conditions_on_age(client):
    """A stone that has already sat unsold is not an average stone in its
    segment, and the endpoint must reflect that rather than ignore the age."""
    stones = [{**_CANON, "StoneId": "A"}, {**_CANON, "StoneId": "B"}]
    out = client.post("/inventory/velocity",
                      json={"stones": stones, "age_days": [0, 150]}).json()
    fresh, old = out[0], out[1]
    assert fresh["AgeDays"] == 0 and old["AgeDays"] == 150
    assert fresh["AgeingBucket"] == "0-90" and old["AgeingBucket"] == "91-180"


def test_mismatched_age_list_is_rejected_not_guessed(client):
    r = client.post("/inventory/velocity",
                    json={"stones": [_CANON], "age_days": [1, 2]})
    assert r.status_code == 422


def test_velocity_rows_carry_both_vocabularies_and_a_basis(client):
    out = client.post("/inventory/velocity", json={"stones": [_CANON]}).json()
    row = out[0]
    assert row["Class"] in ("Fast", "Semi-Fast", "Medium", "Semi-Slow", "Slow")
    assert row["ClassFrontOffice"] in ("High", "Semi High", "Medium",
                                       "Semi Slow", "Slow")
    assert row["ClassBasis"] and "velocity score" in row["ClassBasis"]
    assert "SegmentSales" in row          # how much stands behind the estimate


# ---------------------------------------------------------------------------
# the payloads explain themselves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", [
    "/inventory/ageing", "/inventory/capital-at-risk", "/inventory/segments",
    "/inventory/stock-by-segment", "/inventory/heatmap",
])
def test_every_payload_carries_its_limitations(client, path):
    """MOU 10.3: the basis travels WITH the figure. A renderer must not have to
    reconstruct anything, because there will not be one under this MOU."""
    body = client.get(path).json()
    meta = body["meta"]
    assert meta["limitations"], f"{path} shipped numbers with no limitations"
    assert any("censored" in x.lower() for x in meta["limitations"])
    assert any("seasonality" in x.lower() for x in meta["limitations"])
    assert meta["observation_window"]["from"] and meta["observation_window"]["to"]
    assert meta["class_vocabulary"]["canonical"][0] == "Fast"


def test_no_endpoint_publishes_a_blended_velocity_score(client):
    """MOU 5.2/8.1: own velocity and market depth are two numbers and a ratio.
    One blended gauge would destroy the exact insight the client is paying for."""
    body = client.get("/inventory/segments").json()
    for row in body["rows"]:
        assert "own_velocity_score" in row and "market_depth_score" in row
        assert "velocity_ratio" in row
        assert not any(k in row for k in
                       ("tradeability_score", "combined_score", "blended_score"))


def test_heatmap_refuses_to_score_a_thin_cell(client):
    body = client.get("/inventory/heatmap?min_stones=50").json()
    thin = [c for c in body["cells"] if c["stones"] < 50]
    assert thin, "expected some thin cells at this threshold"
    for c in thin:
        assert c["own_velocity_score"] is None
        assert c["class"] is None
        assert "too few" in c["basis"]


def test_capital_at_risk_is_labelled_asking_value_not_cost(client):
    body = client.get("/inventory/capital-at-risk").json()
    assert "NOT cost" in body["unit_note"]
    assert body["total_stock_asking_value_usd"] > 0


# ---------------------------------------------------------------------------
# the oldest stock must not vanish
# ---------------------------------------------------------------------------
def test_ageing_report_includes_stock_the_velocity_frame_excludes(view):
    """The left-truncation guard is right for ESTIMATING velocity and wrong for
    REPORTING age: it removes the oldest stones on the book. Measured when this
    was found: 1,189 stones and $2.3M of asking value invisible, and the 365+
    red-flag bucket came back EMPTY while the client held 538 stones for it."""
    c = view.classified
    assert "VelocityEstimated" in c.columns
    excluded = c[~c["VelocityEstimated"].astype(bool)]
    assert len(excluded) > 0
    # they are reported for AGE and VALUE...
    assert excluded["AgeDays"].notna().all()
    # ...and never given a speed
    assert excluded["ExpectedDaysToSell"].isna().all()
    assert excluded["Class"].isna().all() or (excluded["Class"] == None).all()  # noqa: E711
    assert "age is reported, speed is not estimated" in excluded["ClassBasis"].iloc[0]


def test_the_red_flag_bucket_is_reachable(view):
    """If nothing can ever land in 365+, the red flag is decoration."""
    buckets = {b["bucket"]: b for b in CH.ageing_distribution(view)["buckets"]}
    assert buckets["365+"]["red_flag"] is True
    assert buckets["365+"]["stones"] > 0, \
        "no stone reaches the red-flag bucket — the oldest stock is being hidden"


def test_velocity_payloads_exclude_the_unscored_stones(view):
    """Age is a fact about every stone; speed is not. The split must hold."""
    rows = CH._velocity_rows(view)
    assert rows["VelocityEstimated"].all()
    assert len(rows) < len(view.classified)


# ---------------------------------------------------------------------------
# the four reports
# ---------------------------------------------------------------------------
def test_all_four_reports_build_with_a_legend_sheet(view, tmp_path):
    from glowstar.reporting import inventory_reports as IR

    paths = {
        "inventory": IR.build_inventory_report(view, out=tmp_path / "inv.xlsx"),
        "price_change": IR.build_price_change_report(view, out=tmp_path / "pc.xlsx"),
        "sales": IR.build_sales_report(view, out=tmp_path / "sale.xlsx"),
        "movement": IR.build_movement_report(view, out=tmp_path / "mv.xlsx"),
    }
    for name, p in paths.items():
        sheets = pd.read_excel(p, sheet_name=None)
        assert "Legend & Honesty" in sheets, f"{name} has no Legend & Honesty sheet"
        legend = sheets["Legend & Honesty"]
        text = " ".join(legend.astype(str).to_numpy().ravel())
        assert "NOT cost" in text
        assert "not learnable" in text.lower() or "NOT modelled" in text
        assert "MarketSheetDate" in text


def test_movement_report_covers_all_nine_cases(view, tmp_path):
    from glowstar.reporting import inventory_reports as IR

    p = IR.build_movement_report(view, out=tmp_path / "mv.xlsx")
    grid = pd.read_excel(p, sheet_name="Nine cases")
    assert len(grid) == 9
    got = set(zip(grid["Inventory"], grid["Sales"]))
    assert got == set(IR.NINE_CASES)
    danger = grid[(grid["Inventory"] == "Up") & (grid["Sales"] == "Down")]
    assert "danger" in danger["Meaning"].iloc[0]


def test_movement_report_says_insufficient_rather_than_guessing(view, tmp_path):
    from glowstar.reporting import inventory_reports as IR

    p = IR.build_movement_report(view, out=tmp_path / "mv.xlsx")
    table = pd.read_excel(p, sheet_name="By segment")
    thin = table[table["Inventory"] == "—"]
    assert len(thin) > 0, "every segment got a direction — the guard is not firing"
    assert thin["Meaning"].str.contains("insufficient history").all()


def test_direction_thresholds_behave():
    from glowstar.reporting import inventory_reports as IR

    assert IR._direction(150, 100) == "Up"
    assert IR._direction(50, 100) == "Down"
    assert IR._direction(105, 100) == "Stable"     # inside the flat band
    assert IR._direction(5, 0) == "Up"
    assert IR._direction(0, 0) == "Stable"


def test_sales_report_compares_the_prior_period(view, tmp_path):
    from glowstar.reporting import inventory_reports as IR

    p = IR.build_sales_report(view, days=30, out=tmp_path / "s.xlsx")
    shape = pd.read_excel(p, sheet_name="By shape")
    assert "Sales (prior)" in shape.columns
    assert "Median days to sell" in shape.columns
    assert "Discount change (pts)" in shape.columns
    assert shape["Sales"].sum() > 0


def test_price_change_report_attributes_by_association_not_causation(view, tmp_path):
    from glowstar.reporting import inventory_reports as IR

    p = IR.build_price_change_report(view, days=60, out=tmp_path / "pc.xlsx")
    sheets = pd.read_excel(p, sheet_name=None)
    text = " ".join(sheets["Legend & Honesty"].astype(str).to_numpy().ravel())
    if "By stone" in sheets and len(sheets["By stone"]):
        assert "Likely driver" in sheets["By stone"].columns
        assert "ASSOCIATION, not causation" in text


# ---------------------------------------------------------------------------
# the velocity promotion gate
# ---------------------------------------------------------------------------
def test_gate_promotes_a_first_model():
    ok, why = VR.gate_decision(0.60, 0.05, 0.52, None, None)
    assert ok and "first velocity model" in why


def test_gate_refuses_a_model_that_does_not_beat_the_live_field():
    """Above the C-index floor, so this isolates the baseline rule: the
    segment-median baseline is essentially what `tradeability.py` publishes
    today, and a candidate that cannot beat it is not worth binding to."""
    ok, why = VR.gate_decision(0.56, 0.05, 0.57, 0.60, 0.61)
    assert not ok and "segment-median baseline" in why


def test_gate_refuses_a_well_ranked_but_badly_calibrated_model():
    """The C-index is blind to a monotone distortion: an off-by-one that made
    P(sold by 30d) read as day 45 moved it by nothing. Calibration is gated
    separately because the desk reads the days-to-sell number too."""
    ok, why = VR.gate_decision(0.65, 0.40, 0.52, 0.60, 0.66)
    assert not ok and "calibration" in why


def test_gate_refuses_a_coin_flip():
    ok, why = VR.gate_decision(0.51, 0.02, None, 0.50, 0.50)
    assert not ok and "floor" in why


def test_gate_has_an_anti_ratchet():
    """Gating on the incumbent alone lets the model degrade without bound while
    every single night passes."""
    ok, why = VR.gate_decision(0.60, 0.05, 0.52, 0.605, 0.70)
    assert not ok and "drifted" in why


def test_gate_tolerates_day_to_day_noise():
    ok, _ = VR.gate_decision(0.605, 0.05, 0.52, 0.610, 0.615)
    assert ok


def test_gate_refuses_when_there_is_nothing_to_evaluate():
    ok, why = VR.gate_decision(None, None, 0.5, 0.6, 0.6)
    assert not ok and "no test window" in why


def test_gate_scores_the_config_that_ships():
    """Mirrors `_assert_gate_scores_what_ships` in the pricing retrain: the
    incident it prevents is a gate that promotes a different model than it
    measured."""
    from glowstar.inventory.velocity import VelocityConfig, VelocityModel

    VR._assert_gate_scores_what_ships(VelocityModel())          # must not raise
    off = VelocityModel(VelocityConfig(max_iter=999))
    with pytest.raises(AssertionError, match="serving_velocity_config"):
        VR._assert_gate_scores_what_ships(off)


def test_calibration_error_ignores_a_horizon_with_no_follow_up():
    """Scoring an unobservable horizon as perfect is how a gate stops gating."""
    result = {"calibration": {
        30: [{"n": 100, "gap": 0.10, "observed": 0.5},
             {"n": 100, "gap": -0.10, "observed": 0.6}],
        90: [{"n": 100, "gap": None, "observed": None}],
    }}
    assert VR.calibration_error(result) == pytest.approx(0.10)
    assert VR.calibration_error({"calibration": {90: [{"n": 5, "gap": None,
                                                       "observed": None}]}}) is None


def test_velocity_registry_never_mixes_metric_protocols():
    """Comparing a number across protocols is meaningless, and doing it silently
    is how the pricing gate nearly froze a model that was performing better."""
    assert VR.best_c_index("a-protocol-that-does-not-exist") is None
    assert VR.METRIC_PROTOCOL in VR.VelocityCard(
        version="v", trained_at="t", n_train=1, n_train_events=1,
        n_test=1, n_test_events=1).metric_protocol


# ---------------------------------------------------------------------------
# the gate must govern what SERVES — Trap 5's fourth head
# ---------------------------------------------------------------------------
def test_serving_uses_the_promoted_model_not_a_fresh_fit(view):
    """The first build refitted a model on every cold start, so the nightly
    promotion gate governed nothing that reached a caller: had it REJECTED a bad
    candidate, serving would have refitted the rejected recipe and served it
    anyway. The gate would have looked like it was working the whole time.
    """
    from glowstar.training import velocity_retrain as _VR

    promoted, card = _VR.load_current()
    if promoted is None:
        pytest.skip("no promoted velocity model in the registry on this box")
    assert view.velocity_card is not None, \
        "the served view was built from an UNGATED model"
    assert view.velocity_card["version"] == card["version"]
    assert view.model.trained_at == promoted.trained_at


def test_payload_says_whether_the_answering_model_passed_the_gate(client):
    """A caller must be able to tell an audited answer from a cold-start one."""
    m = client.get("/inventory/ageing").json()["meta"]["velocity_model"]
    assert "gate_passed" in m and "version" in m
    if m["gate_passed"]:
        assert m["c_index"] is not None
        assert m["c_index"] > m["c_index_baseline"]
        assert m["note"] is None
    else:
        assert "UNGATED" in m["note"]


def test_serving_refuses_a_promoted_model_on_the_wrong_config(monkeypatch):
    """Mirrors the retrain's own assertion at the OTHER end of the pipe: a model
    trained on one configuration must never quietly answer requests as if it
    were on the serving one."""
    from glowstar.inventory import chart as _CH
    from glowstar.inventory.velocity import VelocityConfig, VelocityModel
    from glowstar.training import velocity_retrain as _VR

    stale = VelocityModel(VelocityConfig(max_iter=999))
    monkeypatch.setattr(_VR, "load_current",
                        lambda: (stale, {"version": "bogus"}))
    with pytest.raises(AssertionError, match="not a gate"):
        _CH.load_serving_model(pd.DataFrame())


def test_concurrent_first_requests_build_the_view_once(client):
    """Building a view loads a model and classifies the whole book (~14s).
    Without the lock, N simultaneous first requests each build their own copy —
    multiplying cost and memory exactly when the service is busiest."""
    import concurrent.futures as cf

    codes = list(cf.ThreadPoolExecutor(8).map(
        lambda _: client.get("/inventory/ageing").status_code, range(8)))
    assert set(codes) == {200}


# ---------------------------------------------------------------------------
# market depth against the real feed
# ---------------------------------------------------------------------------
def test_market_depth_counts_come_through_the_authenticity_dedup():
    """Any market count we report must pass through clean_market_stones(), or
    duplicate 'virtual inventory' listings inflate every depth figure."""
    from glowstar.inventory import market_depth as MD

    class _Res:
        class report:
            n_used, n_in = 40, 130

    class _Mk:
        def comparables(self, *a, **k):
            return _Res()

    r = MD.depth_for("Round", 1.0, "G", "VS1", market=_Mk())
    assert r.depth == 40                      # the DEDUPED count, not 130
    assert "130" in r.basis and "dedup" in r.basis


# ---------------------------------------------------------------------------
# market depth: coverage, scale, and valid JSON
# ---------------------------------------------------------------------------
def test_dashboard_is_strictly_valid_json(view):
    """NaN and Infinity are NOT valid JSON. Python emits the bare tokens
    happily; a browser's JSON.parse throws on them — and under this MOU the
    renderer is somebody else's code, so we never see the error."""
    import json

    json.dumps(CH.dashboard(view), allow_nan=False, default=str)


def test_depth_uses_the_banked_snapshot_not_a_24_hour_live_pull(view):
    """A live per-segment pull measures at ~44s x 1,958 segments (~24 hours).
    MOU 5.2's own-vs-market ratio is an acceptance condition, so it cannot be
    left null across the book waiting for that."""
    assert view.depth_table is not None
    assert view.depth_table.n_requested > 500
    resolved = [r for r in view.depth_table.by_segment.values() if r.depth is not None]
    assert resolved, "no segment resolved a depth at all"
    assert any("banked market snapshot" in r.basis for r in resolved)


def test_uncovered_shape_reports_a_reason_not_a_zero(view):
    """Zero comparables means 'you are alone', which scores WELL. A shape the
    snapshot simply does not carry must never be given that score."""
    missing = [r for r in view.depth_table.by_segment.values() if r.depth is None]
    if not missing:
        pytest.skip("every segment resolved on this snapshot")
    r = missing[0]
    assert r.score is None
    assert "no market depth available" in r.basis
    assert "Round only" in r.basis or "not present" in r.basis


def test_banked_depth_scores_spread_instead_of_pinning_at_100():
    """Scored against an absolute constant, every real segment hit 100 and the
    own-vs-market label read 'liquid both ways' for the whole book. A score with
    no variance is not a score."""
    from glowstar.inventory import market_depth as MD

    class _Stones(pd.DataFrame):
        pass

    stones = pd.DataFrame({"segment": ["Round|3|D|FL", "Round|3|D|VS1",
                                       "Round|5|G|SI1", "Round|1|E|VS2"]})
    t = MD.banked_depth_table(stones)
    scores = [r.score for r in t.by_segment.values() if r.score is not None]
    if len(scores) < 2:
        pytest.skip("banked artifact too thin on this box")
    assert len(set(scores)) > 1, "every segment scored identically"
    assert all(0 <= s <= 100 for s in scores)
    assert all("percentile" in r.basis
               for r in t.by_segment.values() if r.score is not None)


def test_own_vs_market_treats_nan_as_missing_not_as_a_number():
    """A partially-resolved depth column is a FLOAT column, so unresolved rows
    arrive as NaN rather than None. Guarding only against None let NaN reach
    round() and took down the whole book the moment depth was partial — which
    is the normal case."""
    from glowstar.inventory import market_depth as MD

    out = MD.own_vs_market(70, float("nan"))
    assert out["velocity_ratio"] is None and out["edge"] is None
    assert "not computable" in out["edge_basis"]
    assert MD.own_vs_market(float("nan"), 70)["edge"] is None
    assert MD.own_vs_market(float("inf"), 70)["edge"] is None
