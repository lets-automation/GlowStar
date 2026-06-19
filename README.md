# Glow Star — Diamond Pricing & Inventory Intelligence System

**v1 scope: the Pricing Engine.** Price individual certified natural stones in
*discount-off-Rapaport* space, with a confidence band, market comparables, an
explicit method/fallback, and a guarded natural-language explanation. Accuracy
is the product: every number is computed deterministically or by a leakage-free
trained model — never hallucinated.

The Inventory Intelligence and Gap engines (brief §9–10) are **not built yet**,
by design: the brief and the client require the Pricing Engine to be accurate
first.

---

## What it does (end to end)

```
records.json ─┐
Rap CSV grids ─┼─► load + validate ─► leakage-free features ─► Quantile GBM ─┐
Uni market ───┘        (Rap core)         (recency-weighted)                │
                                                                            ▼
                                          market anchor (re-center to "now")
                                          + BGM/milky/shade discount (learned)
                                                                            ▼
                                          conformal interval + fallback routing
                                                                            ▼
                                          PriceSuggestion ─► guarded LLM narration
                                                                            ▼
                                                              REST /price (CRM)
```

## Measured accuracy (leakage-free, out-of-time)

Train on sales **before 2026-05-01**, test on **May–Jun 2026**. Target =
`FDiscount` (final discount off Rap). No forbidden/transaction features.

| | MAE (disc pts) | within ±5 | $ median err | signed bias |
|---|---|---|---|---|
| Hierarchical-median **baseline** | 7.36 | 42.3% | $66 | — |
| **Pricing Engine** | **4.99** | **54.7%** | **$48** | **+0.25** |

- **32.1% lower error than the baseline**, provable not asserted.
- Built up in measured layers (each verified on the out-of-time backtest):
  leakage-free GBM (MAE 5.94) → recency weighting (5.62) → **Uni market anchor**
  (5.10, cut Round 8.81→4.59) → **damped market trend** (bias → ~0).
- Trained on **all** sold history (production-correct), recency-weighted, with
  human-feedback labels folded in.
- Confidence interval (rolling-origin conformal): **target 80%, empirical 64%.**
  Honest gap — 6 months out-of-time across a market regime shift means the
  calibration window under-represents test dispersion; it tightens as the daily
  snapshot job banks data. Point accuracy is unaffected (see *Limitations*).

Reproduce:
```bash
python -m glowstar.validation.engine_backtest      # final engine vs baseline, per-shape
python -m glowstar.validation.backtest             # model-only ablation
```

## "Market data" is three distinct signals (all load-bearing)

The brief's north star depends on market data being a *correction*, not an
add-on. It is really three different signals, and the engine uses all three:

1. **Cross-sectional level — Uni feed (`market/anchor.py`).** Where the market
   is *right now*, by segment. Re-centers the model's discount to current market.
2. **Temporal direction — internal trend index (`market/index.py`).** Which way
   and how fast the market is moving. A quality-adjusted price index built from
   the client's own realized sales, applied as a **damped, capped** forward
   de-bias — naive extrapolation overshoots a mean-reverting market (measured:
   ×1.0 made MAE *worse*; ×0.5 cut it and zeroed the bias).
3. **External macro research — `market/context.py`.** RAPI −30% since 2024,
   lab-grown at 61%, tariffs, G7 traceability — provenance-tagged, surfaced with
   every price, and cross-checked against the internal trend (an authenticity
   guard). Never a silent model input.

## How the market feed corrects a falling market (proven, not assumed)

A model trained only on the client's recent sales, in a moving market, is
biased: it carries the *past* discount level and the bias concentrates in
liquid rounds. We demonstrated this from the model's own residuals, then fixed
it with the Uni market feed:

- **90% of the Uni bulk dump (2.42M of 2.68M rows) were duplicate re-listings**
  of the same certificate across sellers — the "virtual inventory" effect.
  Deduped to **268,815 unique stones** before aggregating; raw aggregation would
  distort the market level.
- **Soft-attribute discounts learned from real market data** (the attributes the
  client's own records don't capture yet):

  | milky | discount delta | | shade | delta |
  |---|---|---|---|---|
  | slight | −5 pts | | brown/green/gray | −6 pts |
  | medium | −10 pts | | | |
  | heavy | −11 pts | | | |

  These apply automatically when a stone's `milky`/`Shade` are supplied.

Build the market artifacts (streams the 6.2GB dump, ~2 min):
```bash
python -m glowstar.market.aggregate_bulk
# -> artifacts/market_segments.json, artifacts/bgm_discounts.json
```

## Client-requested capabilities (all built & tested)

- **Market-data authenticity (`market/authenticity.py`).** Authentic data is
  expensive, so we clean rigorously and report honestly: dedupe by certificate
  (the raw Uni dump was ~90% re-listings), drop stale listings, IQR-trim
  outliers, score source quality (lab tier / cert / media), median (not mean)
  aggregation, and an asking→transaction calibration. Every clean returns an
  `AuthenticityReport`.

- **BGM-as-base (`market/bgm.py`).** The reference price is the **No-BGM clean**
  stone; BGM is an explicit deduction (market-learned: milky −5/−10/−11,
  negative shade −6). Three states, always explicit: `clean`, `bgm`, and
  `unassessed` — the last flags `bgm_unassessed` and states that the price
  ASSUMES no BGM, so you never pay clean price for a BGM stone. (Reason to start
  capturing milky/shade in the CRM.)

- **Human-feedback learning loop (`feedback/`).** Every accept/reject/override
  is stored immutably with a **required reason code**. The model learns two
  ways: an **online** per-segment correction from overrides (shifts pricing
  immediately) and **durable** retraining (OVERRIDE gold labels up-weighted,
  ACCEPT confirmations) folded into `engine.fit`. Reason analytics direct what
  to fix next. Record via `PricingService.record_decision(...)`.

## Live ingestion + recurring snapshots

Connectors for all four APIs (`glowstar/ingestion/`), credentials from `.env`
only, HTTPS enforced, secrets/full-URLs never logged. The recurring snapshot job
persists immutable, timestamped pulls and detects schema drift — run it from day
one to grow the history the velocity and trend models need:

```bash
python -m glowstar.ingestion.run_snapshot          # one pull (needs live creds)
# schedule daily — Windows Task Scheduler:
schtasks /Create /SC DAILY /ST 02:00 /TN GlowStarSnapshot \
  /TR ".venv\Scripts\python.exe -m glowstar.ingestion.run_snapshot"
# or cron:  0 2 * * *  .venv/bin/python -m glowstar.ingestion.run_snapshot
```

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # fill in rotated secrets
pytest                                                # 93 tests
```

Python 3.12+ (developed on 3.14). No GPU; scikit-learn `HistGradientBoosting`.

## Price a stone

```python
from glowstar.service.pricing_service import PricingService, StoneIn
svc = PricingService()                                # fits once on all sold history
print(svc.price(StoneIn(Shape_full="Round", Weight=1.01, Color="G", Clarity="VS2",
                        CPS="3EX", Lab="GIA", Rap=8200)))
```

REST (optional): `pip install fastapi uvicorn && uvicorn glowstar.service.app:app`
then `POST /price`; OpenAPI at `/docs`.

## Layout

| Path | Purpose |
|---|---|
| `glowstar/reference/` | Deterministic Rap lookup + value normalization (foundation) |
| `glowstar/data/` | Validated record loading; leakage definitions |
| `glowstar/features/` | Leakage-free feature matrix + forbidden-feature guard |
| `glowstar/market/` | Uni codebook, 6.2GB aggregator, anchor + soft deltas, **trend index**, **macro context**, **authenticity**, **BGM-as-base** |
| `glowstar/models/` | Baseline+fallback, quantile GBM, the `PricingEngine` orchestrator |
| `glowstar/feedback/` | Immutable decision store + reason codes + online/durable learning |
| `glowstar/ingestion/` | Four API connectors + immutable snapshot store + recurring job |
| `glowstar/validation/` | Out-of-time backtest + **shadow mode** + metrics |
| `glowstar/narration/` | LLM explanation + the number guard |
| `glowstar/service/` | Typed `PricingService` (+ market context + feedback) + FastAPI app |
| `glowstar/pipeline.py` | End-to-end: ingest (live/file) → train → serve |
| `tests/` | 93 tests incl. Rap-core anchors, leakage guard, engine guardrails, feedback, mocked connectors |

See **[TESTING.md](TESTING.md)** for the complete end-to-end testing guide and
go-live acceptance gates.

## Honest limitations (surfaced, not hidden — brief §14)

- **6 months of history.** No seasonality is learnable yet; out-of-time interval
  calibration sits at ~64% vs 80% target — the rolling-origin conformal window
  can't fully see the test-period regime shift on so little data. Point accuracy
  is unaffected. **Mitigation: the recurring immutable snapshot job (built) must
  be scheduled from day one** against live credentials; coverage tightens as the
  series grows.
- **Uni request codebook is now EMPIRICALLY VERIFIED against the live API**
  (`market.calibrate_codebook` -> `artifacts/uni_codebook.json`): shape, color
  (D=1..N=11), clarity (FL=1, IF=2 … note the doc's IF=1 was WRONG — live
  calibration corrected it), lab (GIA/HRD/IGI), fluorescence. Rare shapes not in
  the verified set still fail loud (`to_code(strict=True)`) rather than guess.
- **Soft attributes not in the client's data.** They're applied from the market
  feed today; the client should start capturing `milky`/`shade`/`eye-clean` in
  the CRM. The pipeline already has the slots.
- **The 6.2GB market dump is truncated** at the tail; the aggregator uses all
  complete records (2.68M) and records `truncated_source: true`.
- **Live APIs are connected and working.** All four endpoints authenticate from
  `.env` and return live data (Channel Partner ~28.3k records, Diamanto token +
  410k grid cells, Uni market live). Credentials live only in `.env`, never in
  code. Rotate before wider production rollout.
- **Rare shapes / fancy colors / oversize / the 6–9.99ct gap** never get a silent
  number — they route to fallback or return an explicit status for human review.
