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

Train on sales **before 2026-06-01**, test on **June–Jul 2026** (a recent window
that mirrors production: nightly retrain → predict the near term). Target =
`FDiscount` (final discount off Rap). No forbidden/transaction features.

| | MAE (disc pts) | within ±5 | $ median err | signed bias |
|---|---|---|---|---|
| Hierarchical-median **baseline** | 7.19 | 44.4% | $65 | — |
| **Pricing Engine** | **3.88** | **71.8%** | **$33** | **+0.72** |

*(Re-measured 2026-07-10 on the currently promoted model — trained live-rebuilt on
Dec–Jul, with BGM + size-tier + competence guard, split 2026-06-01. NOTE: at the
older, harder 2026-05-01 split — predicting up to 3 months out — the same model
measures MAE 4.28; that stress number is honest too, it is just not what a
nightly-retrained production model does. Historical docs quoted 3.60 / 4.00 on
earlier data/splits.)*

- **46% lower error than the baseline**, provable not asserted. Calibration
  (rolling-origin conformal): target 80%, **empirical 81.2%**. BGM (milky/brown
  from the client's live `BgmComments`) is now a model feature; it improves the
  ~2% milky/brown stones (6.10→5.04 MAE on that subset) and leaves clean stones
  flat — the overall MAE change is within noise (BGM is a per-stone fix, not a
  book-wide one).
- Built up in measured layers (each verified on the out-of-time backtest):
  leakage-free GBM → recency weighting → **segment-aware Uni market anchor**
  (per-segment asking→realized offset, shrunk to global) → **fixed
  `market_month_index`** time feature. The earlier explicit "market-trend shift"
  was **removed**: once the time feature is fixed it double-counts (it had been
  injecting a +bias and breaking interval coverage). Simpler *and* more accurate.
- `recency_half_life` was selected on an **inner validation window (April), never
  the test set** (`glowstar.validation.tune`). The **deployed `anchor_lambda=0.50`
  is a client decision** weighting the live market 50/50 for market-responsiveness;
  inner-validation's MAE-optimal value was 0.25 (better at reproducing *past*
  realized discounts). The engine therefore trades ≈0.3 MAE points of backtest fit
  for tracking the *current* market — a trade validated directly below on live
  client pricing.
- Trained on **all** sold history (production-correct), recency-weighted, with
  human-feedback labels folded in.
- Confidence interval (rolling-origin conformal): **target 80%, empirical 81.4%**
  — well-calibrated (up from 64% before the `market_month_index` fix).

### Forward-pricing validation against live client decisions (strongest evidence)

The backtest above predicts *past realized* discounts. The engine's actual job is
*forward* list pricing. On **99 stones the client re-priced and returned**
(`artifacts/GlowStar_103GS_Priced_SLOTS-Client-Sent.xlsx`), our deployed forward
suggestion matched the client's own discount at:

| Reference vs the client's actual `Disc %` (priced **same-day**) | MAE | within ±5 |
|---|---|---|
| **Our engine suggestion** | **1.93** | **97.0%** |
| The client's own Master grid | 2.95 | 87.9% |
| Raw market asking | 3.36 | 77.8% |

Our forward price tracked the client's real pricing **tighter than their own grid
does**, and much tighter than raw market. NOTE: this 1.93 was measured **same-day**
as the client's decisions; re-pricing the same file today (their `Disc %` is now
~2 weeks old and the live market has moved) measures ~2.6 — the gap is **market
drift, not a model change** (same-day is the fair comparison). One stone,
`OY26-44`, sat ~19 pts off everything — later confirmed genuinely clean (No-BGM),
so its clean price stands and the client's shallower call was a premium judgement.

- **Shadow-mode go-live:** clear the well-tracked shapes; **Sq.Emerald is routed to
  human review in code** (the competence guard defers only shapes measured to lose
  to the segment median at fit time — currently just Sq.Emerald) — the global model prices it
  worse than a naive segment median, so the engine now defers to the segment
  baseline for them rather than auto-pricing (see `PricingEngine` segment guard).

Reproduce:
```bash
python -m glowstar.validation.engine_backtest      # final engine vs baseline, per-shape
python -m glowstar.validation.tune                 # inner-validation tuning (no test leakage)
python -m glowstar.validation.shadow               # per-shape go-live recommendation
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

## Live training, scheduled retrain & a model registry

The model should not be frozen — the market is moving. The retrain job pulls live,
**unions the sold history across every banked snapshot** (the live `GetAllRecord`
serves a rolling window — verified live 19,579 sold vs an earlier 20,143 — so a
single pull silently drops the oldest sales; unioning the immutable snapshots is
what actually *grows* the trainable history), retrains, evaluates leakage-free,
and **promotes the new model only if it matches/beats the incumbent** — so a bad
data day can never silently degrade live pricing. Promoted models are versioned
immutably under `artifacts/models/<version>/` with an accuracy card; serving loads
the gated model instead of retraining on boot.

```bash
python -m glowstar.training.retrain                # pull → assemble → train → GATE → promote
# schedule nightly, after the snapshot pull:
#   0 3 * * *  .venv/bin/python -m glowstar.training.retrain
```

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # fill in rotated secrets
pytest                                                # 104 tests
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
| `glowstar/data/` | Validated record loading; leakage definitions; **snapshot-history assembly** |
| `glowstar/features/` | Leakage-free feature matrix + **whitelist** leakage guard; fixed time feature |
| `glowstar/market/` | Uni codebook, 6.2GB aggregator, **segment-aware anchor** + soft deltas, trend index (direction narration), **macro context (+staleness guard)**, **authenticity**, **BGM-as-base**, **pluggable sources** (Uni; RapNet/IDEX-ready) |
| `glowstar/models/` | Baseline+fallback, quantile GBM, the `PricingEngine` orchestrator, **versioned registry** |
| `glowstar/feedback/` | Immutable decision store + reason codes + online/durable learning |
| `glowstar/ingestion/` | Four API connectors + immutable snapshot store + recurring job |
| `glowstar/training/` | **Nightly retrain with an accuracy promotion gate** |
| `glowstar/validation/` | Out-of-time backtest + **inner-validation tuner** + **shadow mode** + metrics |
| `glowstar/narration/` | LLM explanation + the number guard |
| `glowstar/service/` | Typed `PricingService` (registry-backed) + market context + feedback + FastAPI app |
| `glowstar/pipeline.py` | End-to-end: ingest (live/file) → train → serve |
| `tests/` | 104 tests incl. Rap-core anchors, leakage guard, engine guardrails, feedback, retrain/registry, market sources, mocked connectors |

See **[TESTING.md](TESTING.md)** for the complete end-to-end testing guide and
go-live acceptance gates.

## Honest limitations (surfaced, not hidden — brief §14)

- **6 months of history.** No seasonality is learnable yet; out-of-time interval
  calibration is **81.4% vs 80% target** (up from 64% after the `market_month_index`
  fix). **Mitigation: the recurring immutable snapshot job (built) must be
  scheduled from day one** against live credentials; coverage tightens as the
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
- **Live APIs verified working (2026-06-19).** All endpoints authenticate from
  `.env` and return live data: Channel Partner **28,090 records** (Sold 19,579 /
  Stock 8,511 — a rolling window, see *Live training* above), Diamanto token OK,
  Uni market live (e.g. 482 comps for Round 1.0–1.09 G VS2 GIA). The nightly
  retrain consumes this; serving prefers the gated registry model. Credentials
  live only in `.env` (git-ignored), never in code.
- **Rare shapes / fancy colors / oversize / the 6–9.99ct gap** never get a silent
  number — they route to fallback or return an explicit status for human review.
