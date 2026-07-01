# BUILD BRIEF — Glow Star **Inventory Intelligence + Gap Engine** (Engines 2 & 3)

> This is the build brief for the **second phase** of the Glow Star system. The
> **Pricing Engine (Engine 1) is complete, tuned, tested, and production-wired** —
> do NOT rebuild it. This phase builds the system the client is actually buying:
> turning accurate prices into **inventory intelligence** (fast/slow bifurcation,
> days-to-sell, repricing) and a **gap/assortment engine** (what to acquire).
>
> Read this whole file before writing code. It is self-contained: it assumes the
> reader is a fresh session with no prior conversation, only this repo.

---

## 0. Orientation — what already exists (REUSE IT, don't rebuild)

The repo is a working Python package `glowstar/`. Read these first:

| Path | What it gives you (reuse) |
|------|---------------------------|
| `GlowStar_Build_Prompt.md` | The original Engine-1 brief: domain, data schema, principles. |
| `glowstar/data/loaders.py` | `load_records()`, `sold_stones()`, `stock_stones()`, the verified schema, leakage definitions. **Use these to load data.** |
| `glowstar/data/history.py` | `assemble_sold_history()` — unions sold stones across daily snapshots (the live API serves a rolling window). Use for training data. |
| `glowstar/market/segments.py` | `segment_keys()`, `size_band()`, `SIZE_EDGES` — the canonical segment definitions. **Use the SAME segmentation everywhere.** |
| `glowstar/market/anchor.py` | `MarketTables` (per-segment market medians + BGM). The market-level signal. |
| `glowstar/market/live.py` | `LiveMarket` — live Uni comparables per segment (cleaned). The market-liquidity raw source. |
| `glowstar/market/sources.py` | `get_market_source()` — pluggable feed seam (Uni now; RapNet/IDEX fail loud). |
| `glowstar/models/engine.py` | `PricingEngine` — the fair-price model. The inventory layer consumes its output; it does NOT change pricing math. |
| `glowstar/models/registry.py` | `save_engine()/load_current()/set_current()` + `ModelCard`. **Reuse this exact pattern** to version inventory models. |
| `glowstar/training/retrain.py` | Nightly retrain + **promotion gate**. Mirror this pattern for the velocity model. |
| `glowstar/validation/backtest.py` | `time_split()`, `metrics`. Reuse for out-of-time evaluation. |
| `glowstar/service/pricing_service.py` + `service/app.py` | The serving pattern (pydantic in, dict out) + FastAPI. Add inventory/gap endpoints alongside. |
| `glowstar/narration/narrate.py` + `narration/guard.py` | Template/Claude "why" with a number guard. **Reuse for the gap-engine "why buy it" reasons.** |
| `glowstar/reporting/excel_report.py` | The self-explanatory Excel pattern (Results / Summary / Legend & Honesty sheets). Mirror it for the inventory report. |
| `records.json` (or live pull) | 28k records: **Sold 20,143** (event history) + **Stock 8,185** (the live book to bifurcate) + Transit. |

**Run `pytest` first** to confirm the environment is green (104 tests), then
`python -m glowstar.validation.engine_backtest` to see Engine 1 working.

---

## 1. North star (the client's own words — frame everything around this)

> *Our main focus is not just faster pricing or better margins — it is accurate
> pricing that intelligently bifurcates inventory, so fast-moving and slow-moving
> stock are priced with different strategies, stock turnover increases, and overall
> margin is optimized.*

Pricing accuracy is the INPUT (done). **Inventory intelligence is the OUTCOME the
client is buying.** Engine 2 = bifurcate + reprice for turnover. Engine 3 = tell
them what to hold that they don't.

---

## 2. THE critical idea — separate "our velocity" from "market liquidity"

**Client's explicit nuance:** *a stone not selling in the broad market does not
mean it isn't selling for THEM.* A segment can be illiquid market-wide yet move
fast through Glow Star's own channel/clientele — and vice versa. The system must
NEVER conflate the two. Model them as two distinct signals:

- **Market liquidity** = how active/deep a segment is in the broad market (Uni
  listing counts, days-on-market, dedup'd comparable counts). Source: the market
  layer (`market/live.py`, `MarketTables`).
- **Own velocity** = how fast *Glow Star* actually sells that segment, learned
  from *their own realized sales* (time from listing to sale).

The correct statistical tool to separate them is a **shared-frailty / hierarchical
survival model**: the **baseline hazard = market-driven sell rate**, and a
**per-segment frailty (random effect) = Glow Star's own velocity** relative to it.
Segments where the frailty says "we sell this faster than the market is liquid"
are the client's **edge** — keep stocking even if the market looks slow. (Caveat:
frailty and non-proportional hazards can be hard to identify apart on only 6
months of data — start simple, see §6.)

Deliver BOTH numbers on every segment/stone, plus their ratio, so the client sees
*"market: slow, us: fast → keep it"* explicitly.

---

## 3. Data you have (and the survival setup)

From `records.json` / the live Channel Partner pull (`assemble_sold_history()` for
the union across snapshots):

- **Event/duration fields:** `Status` (Sold/Stock/Transit), `OrderDate` (sale date,
  Sold only), `MarketSheetDate` (listing date), `CreatedDate`, `AvailableDays`,
  `Ageing`. Reconcile `AvailableDays`/`Ageing` against `OrderDate − MarketSheetDate`
  and pick the cleanest duration; document the choice.
- **The survival dataset (build this):**
  - For **Sold** stones: `event = 1`, `duration = days from listing to sale`
    (`OrderDate − MarketSheetDate`, clamp ≥ 0).
  - For **Stock** stones: `event = 0` (RIGHT-CENSORED — they have NOT sold yet,
    they are *not* "never sells"), `duration = today − MarketSheetDate`
    (= `AvailableDays`/`Ageing`).
  - Covariates: shape, size band, color, clarity, CPS, fluorescence, lab, Rap
    band, **and the discount level at listing** (price aggressiveness drives speed)
    — plus the market-liquidity covariate per segment (from the market layer).
- **The live book to act on:** the **Stock** stones (8,185) — these get a velocity
  class + a reprice suggestion + an aging flag.
- **Market liquidity raw signal:** per-segment Uni comparable counts / freshness
  (`market/live.py`, `MarketTables.segments[*]["n"]`), and listing dates for
  days-on-market where available.
- **BGM note (verified):** the Uni feed carries `is_bgm`/`milky`/`shade_name`/
  `eye_clean` per stone, and the engine uses these to learn the BGM *deduction
  values* + clean-base market level. But you **CANNOT** match them to the client's
  individual stones: the Uni live endpoint returns a BLANK `certificate_number`,
  and the bulk dump's certs (`GS…`, 12-digit) don't match the client's 10-digit
  GIA numbers (0 of 28,053 matched). So a client stone's own BGM is unknowable
  from Uni → it stays `unassessed` (priced on the clean base, flagged). The real
  fix is **CRM capture** (milky/shade/eye-clean dropdown). Same for the velocity
  covariate: BGM moves stones slower, but you only get it once captured.

**Leakage rule (same discipline as Engine 1):** when predicting time-to-sell at
listing, do NOT use any post-listing/transaction field as a feature (final price,
final discount, sale date). The duration/label is built from them; they are not
inputs.

---

## 4. ENGINE 2 — Inventory Intelligence

### 4.1 Velocity / days-to-sell (survival model)
- Build a **survival model** for time-to-sale by segment with **right-censoring**.
  - **Baseline (build first, fully explainable):** **Kaplan–Meier** survival curves
    per segment (`lifelines.KaplanMeierFitter`) → median/quantile days-to-sell +
    an interval. Handles censoring correctly out of the box.
  - **Upgrade:** **Cox proportional hazards** (`lifelines.CoxPHFitter`) or a
    **survival GBM** (`sksurv.ensemble.GradientBoostingSurvivalAnalysis` /
    `RandomSurvivalForest`) on the covariates, for per-stone expected days-to-sell.
  - **Own-vs-market separation:** add a **shared frailty / cluster random effect**
    per segment (or `CoxPHFitter` with cluster, or a mixed-effects survival lib)
    so the segment frailty = own velocity vs the market-liquidity baseline (§2).
- **Output per stone & per segment:** expected days-to-sell + interval, a velocity
  score (0–100, higher = faster), the market-liquidity score, and their ratio.
- **Tooling:** add `lifelines` and `scikit-survival` to `requirements.txt`.

### 4.2 Fast / medium / slow bifurcation
- Classify each Stock stone AND each segment as **fast / medium / slow** using the
  velocity score and **aging relative to its segment norm** (a stone aging past its
  segment's median days-to-sell is slowing). Aging buckets: **0–90 / 91–180 /
  181–365 / 365+ days** (industry-standard; >365 = red flag).
- Thresholds **configurable + explainable** (a `Settings`-style dataclass, like
  `config.py`). Every classification carries its basis (velocity score, segment
  median, age, market liquidity).
- Industry benchmark to sanity-check against: jewelry inventory turns ~**0.7–1.2×/
  year**; target ≤ **20% of value in stock > 120 days**. Surface these.

### 4.3 Differentiated repricing (the turnover lever)
- The Pricing Engine gives the **fair market price**. The inventory layer proposes
  a **velocity-adjusted** move on top, by class:
  - **Fast movers:** *reduce the discount / nudge price up toward market* — capture
    margin without killing the sale. Bounded by the pricing engine's confidence band.
  - **Slow / stale movers:** *increase the discount / flag for liquidation* by aging
    bucket — free up capital. Bigger nudge the older it is.
- Show the **projected effect**: expected change in days-to-sell and in margin for
  the suggested move (from the survival model's covariate sensitivity).
- **Never auto-apply** to high-value/low-confidence — flag for human review (reuse
  the Engine-1 thresholds in `config.py`).
- Optimize **GMROI (gross-margin return on inventory), not raw turnover** — margin
  varies by segment, which is exactly the "different strategy per segment" goal.

### 4.4 Inventory chart / dashboard data (endpoints, not a UI)
Provide clean JSON the client's CRM (or a thin dashboard) renders:
- stock by segment; **velocity heatmap** (fast↔slow across shape × size × quality);
- **aging distribution** (the 4 buckets) and **capital tied up in slow movers** ($);
- per-segment: own-velocity vs market-liquidity, expected days-to-sell, suggested
  repricing, projected turnover lift.

### 4.5 Seasonality (honest handling)
- Only ~6 months of history ⇒ **annual seasonality is NOT learnable yet.** Inject
  **trade-calendar priors** (known demand cycles) **clearly labeled as priors**,
  and design so the model replaces them with learned seasonality as history passes
  a year. Do not present priors as data-derived (same rule as `market/context.py`).

---

## 5. ENGINE 3 — Gap / Assortment Engine

**Purpose:** recommend stones to **acquire or manufacture** that are NOT in stock
but that the market + the client's own velocity say they should hold — with the
reason attached ("why buy it").

- **v1 (build now):** cross-reference **high own-velocity × high market-liquidity ×
  healthy expected margin** segments against **current stock coverage**. Where Glow
  Star is **under-stocked** in a segment that (a) they sell fast and (b) the market
  shows real demand/depth → recommend acquisition, with: segment spec (shape/size/
  color/clarity band), the why, expected days-to-sell, expected discount/margin
  (from the Pricing Engine), and a **confidence level**.
- **v2 (design for, don't block on):** ingest **lost-inquiry data** (customers who
  asked for stones not in stock) once the CRM captures it — the strongest demand
  signal. **Recommend the client start capturing lost inquiries now.**
- **"Why buy it" reasons:** reuse `narration/narrate.py` (+ the number guard). LLM
  optional; template is clear and hallucination-free by construction.

---

## 6. Build order (phased, each with an acceptance gate)

- **Phase A — Survival foundation.** Build the censored survival dataset (§3) +
  unit tests proving censoring is correct (Stock = censored, not "never sells").
  Kaplan–Meier per segment with intervals. *Gate:* per-segment median days-to-sell
  reproduces, censoring verified, sanity vs `AvailableDays`.
- **Phase B — Velocity model + own-vs-market.** Cox/survival-GBM with covariates;
  add the segment frailty / market-liquidity covariate so own-velocity and
  market-liquidity are separated and both reported. *Gate:* out-of-time concordance
  (C-index) beats a naive segment-median baseline; calibration checked.
- **Phase C — Bifurcation + repricing.** Fast/medium/slow classes; velocity-adjusted
  reprice bounded by the Engine-1 confidence band; projected days-to-sell/margin
  effect; human-review flags. *Gate:* shadow-mode style review — suggestions are
  sensible on the live Stock book; no auto-apply on high-value.
- **Phase D — Inventory chart endpoints + Excel report.** The dashboard JSON (§4.4)
  + a self-explanatory Excel (mirror `reporting/excel_report.py`) with a Legend &
  Honesty sheet. *Gate:* a non-technical reader understands every column.
- **Phase E — Gap engine + service + retrain.** v1 market-driven gap recs with
  reasons; REST endpoints; a nightly retrain of the velocity model with a
  promotion gate (mirror `training/retrain.py`); registry-versioned. *Gate:* CRM-
  ready endpoints documented (OpenAPI); audit log; honest limitations surfaced.

---

## 7. Architecture (new packages; reuse everything else)

```
glowstar/
  inventory/
    survival.py     # censored dataset builder + KM + Cox/survival-GBM + frailty
    velocity.py     # velocity score, days-to-sell, own-vs-market separation
    bifurcate.py    # fast/medium/slow classification + aging buckets
    reprice.py      # velocity-adjusted repricing (bounded by Engine-1 CI)
    chart.py        # dashboard JSON (heatmap, aging, capital-at-risk)
  gap/
    assortment.py   # under-stocked × high-velocity × liquid → buy/make recs + why
  service/          # ADD inventory + gap endpoints to the existing FastAPI app
  reporting/        # ADD inventory_report.py (mirror excel_report.py)
  validation/       # ADD survival_backtest.py (C-index, calibration, out-of-time)
  training/         # ADD velocity retrain with a promotion gate (mirror retrain.py)
```

Reuse `config.py`, `data/*`, `market/*`, `models/registry.py`, `narration/*`,
`feedback/*`. Add `lifelines`, `scikit-survival` to `requirements.txt` (pinned).

---

## 8. Non-negotiable principles (same as Engine 1 — apply to every module)

1. **Numbers are computed, never hallucinated.** Velocity, days-to-sell,
   probabilities, capital-at-risk are deterministic arithmetic or a trained model.
   The LLM only narrates already-computed values (number guard enforced).
2. **Every prediction ships with a confidence band and a basis** (interval, the
   number of comparable/at-risk stones, which method produced it).
3. **Right-censoring is respected.** Unsold stock is censored, never "never sells."
   Validating without it is invalid.
4. **Human-in-the-loop** for high-value / low-confidence / rare segments. No
   auto-apply there.
5. **Auditability.** Every recommendation stores inputs, model version, basis,
   timestamp (reuse the registry + an audit log).
6. **Surface limitations honestly** (§9), never bury them.
7. **Own-velocity and market-liquidity are reported separately, never merged.**

---

## 9. Known constraints — STATE these in outputs, don't hide them

- **6 months of history.** No annual seasonality learnable; slow-mover days-to-sell
  is right-censored and starts wide, tightening as the daily snapshot job banks
  data. Use trade-calendar priors (labeled).
- **Rare shapes / big stones are statistically thin** — fallback to the coarsest
  segment with support + human review (mirror Engine-1's hierarchical fallback).
- **Lost-inquiry / true demand data does not exist yet** — build the market-driven
  gap engine now; advise the client to start capturing lost inquiries.
- **Frailty vs non-proportional hazards** can be hard to identify on thin data —
  start with KM + a simple market-liquidity covariate; add frailty as data grows.
- **Market feeds are asking-price + duplicated** ("virtual inventory") — keep using
  the authenticity dedup (`market/authenticity.py`).

---

## 10. Data-capture recommendations to give the client (raise these early)

These unlock the biggest future gains and cost the client almost nothing now:
1. **BGM/soft attributes** (milky, shade, eye-clean) per stone in the CRM — the #1
   pricing-accuracy lever (the pipeline slots already exist). This is the ONLY way
   to get it: it is NOT on GIA/IGI certs and cannot be matched from Uni (proven).
   Also a velocity covariate (BGM stones move slower).
2. **Lost inquiries** (stones customers asked for but weren't in stock) — the
   strongest demand signal for the Gap engine.
3. **Rejection reasons** on every overridden/rejected suggestion — already wired in
   Engine 1 (`feedback/`); make sure the CRM surfaces the reason-code dropdown.

---

## 11. Definition of done

- Censored survival dataset + KM baseline + velocity model (Cox/survival-GBM),
  out-of-time validated (C-index beats baseline; calibration reported).
- Own-velocity vs market-liquidity reported separately on every segment/stone.
- Fast/medium/slow bifurcation + aging buckets + velocity-adjusted repricing
  (bounded by Engine-1 CI, human-review flags, projected days-to-sell/margin).
- Inventory chart endpoints (heatmap, aging, capital-at-risk) + a self-explanatory
  Excel report (Legend & Honesty sheet).
- Gap engine v1: market-driven acquire/manufacture recs with reasons + confidence.
- Velocity model registry-versioned with a nightly promotion-gated retrain.
- Service API documented (OpenAPI); audit log; honest limitations surfaced in
  outputs. Tests green.

*Build for correctness and auditability first; speed and polish second. When in
doubt between a convenient shortcut and a verifiably-correct path, take the correct
path and document the tradeoff. Reuse Engine 1 — do not duplicate it.*
```
