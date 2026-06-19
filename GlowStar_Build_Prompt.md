# BUILD BRIEF — Glow Star Diamond Pricing & Inventory Intelligence System

---

## 0. Reference materials provided with this brief

The following files accompany this brief. Read them before building.

| File | What it is |
|------|-----------|
| `records.json` | Real production dump from the inventory/sales API — 28,408 records, 30 fields. This is the ground-truth schema and the training/backtest data. |
| `CSV2_ROUND_8_4.csv` | Rapaport price grid snapshot (Round), dated 24-Apr-2026. Columns: shape, clarity, color, minSize, maxSize, price/ct, date. |
| `CSV2_PEAR_8_4.csv` | Rapaport price grid snapshot (Pear). In the trade, the Pear/fancy list prices **all** fancy shapes. |
| `Rap_history__1_.csv` | The client's **internal** price grid history (NOT Rapaport's list). 3 snapshots: Mar-2025, Aug-2025, Mar-2026. Coarse brackets. Optional cross-reference only. |
| `API-Documentation.docx` | The four live API endpoints and their auth. Credentials inside it must NOT be hardcoded (see §11 Security). |

---

## 1. Mission and north star

The client ("Glow Star", natural diamonds only — no lab-grown) wants an AI system that integrates into their existing CRM to do two things: **price individual polished stones accurately**, and **bifurcate inventory into fast-moving vs slow-moving segments so each is priced with a different strategy** to raise stock turnover while optimizing margin.

**North star (the client's own words, paraphrased):** the goal is *not* faster pricing or higher margin in isolation — it is *accurate pricing that drives intelligent inventory bifurcation, so fast and slow movers are handled with different strategies, turnover rises, and overall margin is optimized.* Build and frame the system around this. Pricing accuracy is the input; inventory intelligence is the outcome they are buying.

**Scope for v1:** individual certified stones only. Melee / parcel (sieve-graded, sold by lot) pricing is explicitly **out of scope** for v1 per the client.

---

## 2. Non-negotiable principles (apply to every module)

1. **Numbers are computed, never hallucinated.** Every price, discount, day-count, and probability is produced by deterministic arithmetic or a trained model. A language model may *explain* a number; it may never *invent* one. The LLM layer receives already-computed values and narrates them. If a value is missing, it says so — it does not fill the gap.
2. **Every prediction ships with a confidence band and a basis.** No bare point estimate. Output the estimate, an interval, the number of comparable stones it was derived from, and which method produced it (model / segment-median / fallback).
3. **Validation is leakage-free or it is invalid.** Use out-of-time splits (train on older sales, test on newer). Never let test-period information into training. Specifically, **do not train the price model on `BasePriceDiscount` or `Discount`** — these are human pricing decisions on the same stone and leak the answer (see §7.4). They may be *displayed* for comparison, never used as model inputs for the honest accuracy claim.
4. **Human-in-the-loop for low-confidence and high-value.** Any stone where the model confidence is low, the comparable count is small, the shape is rare, or the value is high (configurable threshold, e.g. ≥ $50k) is flagged for human review rather than auto-applied.
5. **Auditability.** Every suggestion stores its inputs, model version, comparables used, and timestamp. The client must be able to reconstruct *why* any number was produced.
6. **Surface limitations, don't bury them.** The system honestly reports when it is extrapolating (rare shape, thin data, stale market). See §10.
7. **Security first.** No plaintext credentials in code or repo. See §11.

---

## 3. Data sources — the four APIs

All four are documented in `API-Documentation.docx`. **Credentials are supplied via environment variables / a secrets manager, never hardcoded.** The values currently in the doc are to be treated as compromised and rotated before production (see §11).

### 3.1 Auth token (Diamanto)
- `POST https://pricingapi.diamanto.co/api/token`
- Body: `x-www-form-urlencoded` (`username`, `password`, `grant_type=password`, `client_id=ngAuthApp`, `udid`). Credentials from env.
- Returns `access_token` used as a Bearer token for the pricing-history endpoint.

### 3.2 Internal price-grid history (Diamanto `GetCellsHistory`)
- `POST https://pricingapi.diamanto.co/api/SpreadSheetCells/GetCellsHistory`
- Auth: Bearer `access_token`. Body: `{ "fromDate": "...", "toDate": "..." }`.
- Returns the **client's own pricing team's** historical grid (article-wise). This is *their* reference sheet, **not** the Rapaport list. Use as an optional secondary reference frame only; the primary base price is the `Rap` field already attached to each stone (§4).

### 3.3 Market comparables (Uni Diamonds)
- `POST https://app.uni.diamonds/api/1.03/free-form-price-analysis/export-report/1`
- Auth: headers (`deviceid`, `token`, `platform`) from env.
- Body: `form-data` with filters: `shape`, `size_from`, `size_to`, `country_id[]`, `color[]`, `clarity[]`, `lab_ids[]`, `fluorescence_intensity[]`, etc.
- **This is the market/comparables feed.** The client confirmed RapNet data is NOT available to us — Uni Diamonds is the live market source. Used for the comparables panel (§7.5) and the gap engine (§9).
- **CONFIRM — code mappings.** The filter values are numeric codes (`shape:1`, `color[0]:1`, `clarity[0]:1`, `lab_ids[0]:1`, `fluorescence_intensity[0]:7`). The doc shows examples only. Obtain/confirm the full code→value mappings for shape, color, clarity, lab, fluorescence, country before relying on this feed. Build a mapping module with the confirmed tables; fail loudly on unmapped codes.

### 3.4 Inventory + sales (Channel Partner `GetAllRecord`)
- `GET https://channelpartnerapi.azurewebsites.net/api/ChannelPartner/GetAllRecord/{user}/{pass}`
- Returns **all** records. Sales and stock are distinguished by `Status` (§4). This single endpoint is the source for both the training data and live inventory.
- **No documented pagination/filtering.** The full pull returned ~28k records (~22 MB) in one response. Handle as a full snapshot; build for the possibility that they later add paging.

### 3.5 Ingestion requirements (build immediately)
- **Recurring snapshot job from day one.** Schedule a daily (or at minimum weekly) `GetAllRecord` pull and persist each snapshot, timestamped and immutable. Reason: the client has only **6 months** of history (§10). Every snapshot banked from now grows the time series the inventory/velocity models need. This is cheap now and impossible to recover later. Do not skip it.
- Normalize and load each source into the data layer (§6). Detect and log schema drift between snapshots.
- Idempotent loads keyed on `StoneId` + snapshot date.

---

## 4. Canonical data schema (VERIFIED from `records.json`)

28,408 records, 30 fields. Status split: **Sold 20,143 / Stock 8,185 / Transit 80.**

| Field | Type | Meaning / notes |
|-------|------|-----------------|
| `StoneId` | str | Unique stone identifier (e.g. `JQ25-62`). Primary key. |
| `Shape` / `Shape_full` | str | Code (`RBC`, `PB`, …) and full name (`Round`, `Pear`, …). **27 distinct shapes** (§4.2). |
| `Weight` | float | Carat. Range observed 0.15 – 31.05. |
| `Color` | str | D–I observed (and lower). |
| `Clarity` | str | IF, VVS1…I3 (incl. SI3). |
| `Fluorescence` | str | `Non`, `Fnt`, `Med`, `Stg`, `Vsl`, `Slt`, `Vstg`. |
| `CPS` | str | Cut/Polish/Symmetry combined, e.g. `3EX`, `EX`, `VG`, `GD`, `VG-GD`, `FR`. ~2 missing. |
| `Lab` | str | `GIA` (91%), `IGI`, `HRD`, `NC` (no cert), `None`. |
| `CertificateNo` | str | Cert number. ~331 missing (the uncertified stones). |
| `Location` | str | Stock location (e.g. Belgium). |
| `Status` | str | **`Sold` / `Stock` / `Transit`.** Drives the sold-vs-inventory split. |
| `Ostatus`, `LeadStatus` | str | Order/lead status (e.g. `Delivered`). |
| `Rap` | float | **Rapaport base price $/ct, attached per stone. 100% populated, never zero. Frozen as-of-sale for sold stones (§4.1).** This is the base denominator — you do NOT need external Rap sheets to compute discount. |
| `BasePriceDiscount` | float | Discount % off the client's **internal** base sheet. (Display/compare only — do NOT use as a model input; see §7.4.) |
| `Discount` | float | Current discount % off `Rap`. (Human decision — do NOT use as a model input.) |
| `NetAmount` | float | Current market price of the stone. For sold stones it is effectively frozen at sale (§4.1); for stock it is today's market price. |
| `PerCarat` | float | Current price per carat. |
| `AvailableDays` | int | Days the stone was/has been available. ~1,711 zero. |
| `Ageing` | int | Aging in days. ~1,132 zero. |
| `OrderDate` | str | **Sale date. Populated only for `Sold`.** |
| `MarketSheetDate` | str | Listing/entry date in the market sheet. |
| `CreatedDate` | str | Record creation date. |
| `FAmount`, `FPerCarat`, `FNetAmount` | float | **Final** sale amount / per-carat / net. **Populated only for `Sold`** (8,265 blank = the unsold). |
| `FDiscount` | float | **Final discount % off `Rap`. This is the PRIMARY MODELING TARGET.** Populated only for `Sold`. |
| `IsDelivered`, `IsRejected` | bool | Fulfillment flags. |

### 4.1 Verified relationships (rely on these)
- **Exact identity, all 20,143 sold stones, zero mismatches:**
  `FNetAmount == Rap * (1 + FDiscount/100) * Weight`.
  The dataset is internally consistent. Use this identity for validation and to convert between discount-space and price-space.
- **`Rap` is frozen as-of-sale** for sold stones (evidence: the identity above holds exactly, and `NetAmount==FNetAmount` agreement is flat at 89–94% across all sale months with no decay over time — it would decay if values were refreshed to "today"). Therefore historical discounts are correct as stored; **no external historical Rap reconstruction is needed.**
- **`FDiscount ≈ Discount` for ~92% of sold stones**; the ~8% gap is the small negotiation/closing delta between listed and final. Model the target `FDiscount`; the delta itself can be a later refinement.
- **`BasePriceDiscount` is consistently more negative than `Discount`** (discount off the internal sheet, whose values run higher than Rap).

### 4.2 Shape distribution (design implication)
Round 17,683 dominates; substantial volume in Oval, Pear, Heart, Emerald, Marquise, Princess; then a **long tail of rare fancies** (Cushion variants, Kite, Hexagonal, Old Miner, Rosecut, Trilliant, etc. — some with a single example).
- Rapaport publishes a Round list and one fancy (Pear) list that prices all fancies. The `Rap` field already encodes this per stone.
- **ML implication:** common shapes train well; rare shapes cannot be modeled statistically and MUST route to the attribute-based fallback with wide confidence + mandatory human review (§7.6). Build this from the start.

### 4.3 Data quality (verified)
- Core fields ~100% complete. `F*` fields and `OrderDate` blank only for the 8,265 unsold (correct).
- **Outliers: 19 of 20,143 (0.09%)** have `FDiscount` > 0 (premium/likely error) or < −90. Filter/winsorize before training; log them, don't silently drop in production.
- Uncertified (`NC`/`None`, ~349) and big rare stones (6–9.99ct: 8; ≥10ct: 6) are sparse — special handling, not statistics.

---

## 5. Domain knowledge the system must encode

- **Discount-off-Rap is the trade's pricing language.** Stones are priced as a percentage off the Rapaport base. The system reasons and predicts in discount-space, then converts to price via the §4.1 identity.
- **The 4Cs drive value:** Carat (weight, with bracket cliffs), Color (D best → down), Clarity (IF → I3), Cut (here folded into `CPS`). These plus shape, fluorescence, and lab are the primary features.
- **Bracket cliffs.** Rapaport prices jump at size boundaries (e.g. 0.99→1.00ct, 1.49→1.50ct). A stone just over a boundary is worth materially more per carat. Encode bracket membership AND exact weight as features; never interpolate across a cliff naively.
- **The 6–9ct gap.** The Round Rapaport CSV jumps from 5.00–5.99 straight to 10.00–10.99 — there is no published cell for 6.00–9.99ct; these price by negotiation/interpolation. The client's `Rap` field already supplies a value per stone, but any *independent* Rap lookup must handle this gap explicitly (special-case, not crash, not silent wrong value).
- **Lab hierarchy.** GIA generally commands a premium and tighter pricing vs IGI/HRD for equivalent grades; uncertified (`NC`) is a different regime. Treat `Lab` as a meaningful feature, not noise.
- **Fluorescence** affects value non-trivially (often a discount in colorless/near-colorless, sometimes neutral lower down). It's in the data — use it.
- **Soft attributes NOT yet in the data but material to price** (the next accuracy lever): eye-clean (vs included), BGM (Brown/Green/Milky tinge), shade/overtone, milkiness, make nuance beyond CPS. These explain residual price variance the current fields can't. **Action:** design the schema and feature pipeline so these slot in cleanly when available, and recommend the client begin capturing them in the CRM. Do not block v1 on them.
- **Lost-inquiry / demand data** (stones customers asked for but weren't in stock) is the ideal input for the gap engine (§9) and does not exist yet. Build the market-driven version now; design for lost-inquiry ingestion later.

---

## 6. Architecture

Build a new backend service (the client will integrate its outputs into their existing self-managed CRM via API). Keep it modular so each engine is independently testable and deployable.

**Layers:**
1. **Ingestion** — connectors for the four APIs (§3), the recurring snapshot scheduler, normalization, and schema-drift detection.
2. **Data layer** — a persistent store (relational DB recommended, e.g. PostgreSQL) holding: current inventory, immutable historical snapshots, sales records, Rapaport reference grids, the internal grid history, and a cached/append-only store of Uni market pulls. Version everything by snapshot date.
3. **Feature & reference layer** — deterministic Rap lookup (with bracket + 6–9ct handling), size-bracket normalization across the differing schemes (the Rap CSVs use `1.00`; the internal grid uses `1`, and lumps `5–9.99`/`10–99` — normalize), feature builders, soft-attribute slots.
4. **Model layer** — the three engines (§7–9), each versioned, each emitting estimate + confidence + basis.
5. **LLM narration layer** (§8.5 of engine 1, reused) — Anthropic Claude API (configurable model) to turn computed outputs into human explanations and the comparables/why narrative. Numbers are passed in, never generated.
6. **Service API** — clean REST/JSON endpoints the client's CRM calls: price a stone, get inventory intelligence, get gap recommendations, get the explanation for a suggestion. Documented (OpenAPI).
7. **Observability** — logging, audit trail, model-drift monitoring, data-freshness alerts.

**Stack:** Python is the natural choice (data + ML). Use a typed, tested codebase. Pin dependencies. Provide a reproducible environment (e.g. `pyproject.toml`/`requirements.txt` + Makefile). Do not over-engineer infra in v1; correctness and testability first.

---

## 7. ENGINE 1 — Pricing Engine

**Purpose:** for any stone (in stock or hypothetical), produce a recommended discount-off-Rap and resulting price, with a confidence band and supporting comparables.

### 7.1 Target and conversion
- **Target:** `FDiscount` (final discount % off Rap), learned from the 20,143 sold stones.
- Convert prediction to price with the verified identity: `price = Rap * (1 + predictedDiscount/100) * Weight`.

### 7.2 Deterministic Rap lookup core (build first, independently testable)
- Implement a function: given shape, weight, color, clarity → Rapaport $/ct, using the Round/Pear CSV grids (Pear list for all fancies). Bracket-aware. **Explicitly handle the 6–9ct gap** and out-of-grid sizes (e.g. >10.99ct) with a documented rule, not a silent default.
- Normalize the two bracket/size-string schemes (CSV `1.00` vs internal `1`, lumped bands).
- This module must be 100% deterministic and unit-tested against known cells from the CSVs (e.g. Round D/IF 1.00–1.49 = 15000; the 10.00–10.99 cell = 140000/ct). It is the foundation everything else trusts.

### 7.3 Features (model inputs)
Allowed: `Shape_full`, `Weight` (+ bracket membership), `Color`, `Clarity`, `CPS`, `Fluorescence`, `Lab`, `Location`, `Rap` (base), time features, and the **Uni market comparables** summary (§7.5). Add soft attributes (§5) when available.

### 7.4 Forbidden features (leakage — critical)
**Never train on `BasePriceDiscount`, `Discount`, `NetAmount`, `PerCarat`, `FAmount`, `FPerCarat`, `FNetAmount`, or anything derived from the final transaction.** These encode the human's pricing decision or the answer itself. They may be shown alongside the model output for comparison, but including them invalidates the accuracy claim. (A first-pass test that *did* include `BasePriceDiscount` scored ~3.8 pts MAE; the honest, leakage-free number must be measured without it and reported as such.)

### 7.5 Comparables panel (Uni Diamonds)
- For a given stone, query Uni (§3.3) for similar listings (matching shape/size band/color/clarity/lab/fluorescence). Return the distribution of market asking prices/discounts (count, min, median, max).
- Show this to the human pricer and feed a summary statistic into the model. It also bounds the suggestion and powers the "why this price" narrative.

### 7.6 Rare-shape & sparse fallback
- If the stone's shape/segment has too few training examples (configurable, e.g. < 30), route to a hierarchical fallback: predict from the coarsest segment with enough data (shape→size→color→clarity hierarchy), widen the confidence band, mark `method = fallback`, and require human review.

### 7.7 Outputs
For each stone return: `suggestedDiscount`, `suggestedPricePerCarat`, `suggestedNetAmount`, `confidenceInterval` (low/high), `comparableCount`, `method` (model/fallback), `marketComparables` summary, and `flags` (low-confidence / high-value / rare-shape / stale-market). Plus a natural-language explanation from the LLM layer (§8 below — reused).

### 7.8 Confidence bands
Produce genuine intervals (quantile regression, conformal prediction, or model-ensemble spread). Calibrate them: on held-out data, the stated X% interval should contain the truth ~X% of the time. Report calibration in the backtest.

---

## 8. The LLM narration layer (used by all engines)

- Use the Anthropic Claude API (model configurable) to generate explanations: why this discount, how it compares to market, what's driving it (e.g. "G/VS2 3EX GIA round, fluorescence none; market comparables cluster at −52% to −56%; suggested −53% reflects strong cut and clean clarity").
- **Hard rule:** the LLM receives the already-computed numbers and comparables as structured input and only narrates them. It must not produce or alter any figure. Validate LLM output: if it emits a number not present in the structured input, reject/regenerate.
- Keep prompts deterministic and structured (pass JSON of computed facts; request a bounded explanation). Log prompt + response for audit.

---

## 9. ENGINE 2 — Inventory Intelligence Engine (the north-star deliverable)

**Purpose:** classify the live book into fast/slow movers and drive differentiated pricing strategy to lift turnover while protecting margin. This is what the client is actually buying — give it primacy.

### 9.1 Velocity & days-to-sell
- Compute realized days-to-sell from sold stones (`OrderDate` − `MarketSheetDate`/`CreatedDate`; `AvailableDays`/`Ageing` are also provided — reconcile and use).
- Build a **survival model** for time-to-sale by segment (shape/size/color/clarity/price-band). Handle **right-censoring** correctly: stock not yet sold is censored, not "never sells." Kaplan–Meier by segment as a baseline; a survival GBM/Cox model as the upgrade.
- Output per stone/segment: expected days-to-sell + interval, and a velocity score.

### 9.2 Fast/slow bifurcation
- Classify each stock stone and each segment as fast / medium / slow mover using velocity score and aging relative to segment norms.
- For each class, a differentiated strategy:
  - **Fast movers:** reduce discount / raise price toward market — capture margin without killing turnover.
  - **Slow / stale movers:** increase discount, tighten, or flag for liquidation — free up capital. Use aging buckets.
- Make thresholds configurable and explainable. Every reprice suggestion carries its reason and confidence.

### 9.3 Repricing engine
- For each stock stone, suggest a price/discount adjustment given its class, current market (Uni), aging, and the pricing model. Show projected effect (e.g. expected change in days-to-sell). Never auto-apply to high-value/low-confidence without review.

### 9.4 Inventory chart / flow
- Provide the data and endpoints for an inventory chart: stock by segment, velocity heatmap (fast↔slow across shape×size×quality), aging distribution, capital tied up in slow movers. This is the operational dashboard the client asked for.

### 9.5 Seasonality (honest handling)
- With only 6 months of history, seasonality cannot be learned. Inject it as **trade-calendar priors** (e.g. known demand cycles) clearly labeled as priors, and design so the model replaces them with learned seasonality as history accrues past a year. Do not present priors as data-derived.

---

## 10. ENGINE 3 — Gap / Assortment Engine

**Purpose:** recommend stones to acquire or manufacture that are NOT in stock but that the market/velocity says the client should hold, with a reason.

- **v1 (build now):** cross-reference high-velocity, high-liquidity segments (from §9 velocity + Uni market demand/liquidity signals) against current stock coverage. Where the client is under-stocked in a segment that sells fast and shows market demand, recommend acquisition/manufacture, with the reasoning (velocity, market depth, expected days-to-sell, expected discount/margin).
- **v2 (design for, don't block on):** ingest lost-inquiry data once the CRM captures it — customers who asked for stones not in stock are the strongest demand signal. Recommend the client start capturing lost inquiries now.
- Every recommendation: segment spec (shape/size/color/clarity band), the why, expected economics, and confidence.

---

## 11. Security & operations

- **No secrets in code or repo.** All credentials (Diamanto user/pass, Uni token, Channel Partner user/pass) via environment variables or a secrets manager. Provide a `.env.example` with variable names only.
- Treat the credentials currently in `API-Documentation.docx` as **compromised**; the build assumes they will be rotated and re-delivered securely. Flag this to the client.
- The Channel Partner endpoint passes credentials in the URL path — confirm HTTPS end-to-end and avoid logging full URLs.
- Audit log every suggestion (inputs, model version, comparables, output, timestamp, user).
- Monitor: data freshness (snapshot age), model drift (prediction error vs incoming actuals), Uni feed health, and unmapped-code errors.
- Persist immutable historical snapshots (§3.5).

---

## 12. Validation, backtesting & accuracy framework (the credibility core)

This is how the client is convinced and how go-live is gated. Implement it as a first-class, repeatable harness.

- **Out-of-time split.** Train on earlier sales, test on the most recent window (e.g. train ≤ Apr-2026, test May–Jun-2026). No test-period leakage. No forbidden features (§7.4).
- **Metrics, reported overall AND per shape/size/quality segment:**
  - MAE and median absolute error in **discount points**.
  - Error in **dollars per stone** (median and mean).
  - Share of stones within **±3** and **±5** discount points.
  - Confidence-interval calibration (does the X% band contain truth ~X% of the time?).
- **Baseline vs model.** Always report a transparent baseline (hierarchical median discount by segment) alongside the model, so improvement is provable, not asserted.
- **Shadow mode before go-live.** Run the system silently in parallel with the client's pricers for a defined period; compare suggestions to human decisions and to realized sales. Go live only on segments where the system matches or beats the human within an agreed tolerance. Keep humans in control elsewhere.
- **Acceptance thresholds** to be agreed with the client per segment (do not hardcode a single global number; rare/high-value segments warrant stricter review).

---

## 13. Build order (phased — each phase has an acceptance gate)

**Phase 0 — Foundation.** Ingestion for all four APIs; data layer + immutable snapshots + scheduler; deterministic Rap lookup core (§7.2) fully unit-tested; schema validation against §4. *Gate:* clean load of `records.json` equivalents, Rap core passes all tests, recurring pull running.

**Phase 1 — Pricing Engine + backtest.** Features, leakage-free model, confidence bands, comparables panel, rare-shape fallback, full validation harness (§12). *Gate:* leakage-free backtest meets agreed accuracy thresholds per segment; baseline beaten; intervals calibrated.

**Phase 2 — Inventory Intelligence + shadow mode.** Velocity/survival model, bifurcation, repricing, inventory chart endpoints; shadow-mode comparison running. *Gate:* shadow-mode results acceptable on target segments.

**Phase 3 — Gap Engine + gated go-live + CRM integration.** Market-driven gap recommendations; documented service API; integrate outputs into the client CRM; go live only on cleared segments. *Gate:* CRM integration verified, audit + monitoring live, agreed segments in production.

---

## 14. Known constraints — the system must state these, not hide them

- **6 months of sales history (Dec-2025 → Jun-2026, verified).** No annual seasonality is learnable yet; slow-mover days-to-sell is right-censored and starts as a wide estimate that tightens as history grows. Mitigate with the recurring snapshot pull and trade-calendar priors.
- **Rare shapes & big stones are statistically thin** — fallback + human review, never silent extrapolation.
- **Soft attributes (eye-clean, BGM, shade, milkiness) are not yet captured** — they are the next accuracy lever; design for them and advise the client to start collecting.
- **`NetAmount` is today's market for *stock*** — correct for live valuation, but never use it to compute *historical* discounts (use the frozen `Rap`/`FDiscount` instead).
- **Uni code mappings unconfirmed** — block that feed behind confirmed mappings.

---

## 15. Definition of done

- All four connectors working against live APIs (with rotated secrets), recurring snapshots persisting.
- Deterministic Rap core: 100% test pass, including 6–9ct and out-of-grid cases.
- Pricing engine: leakage-free backtest reproducible via one command, metrics reported per segment with calibrated intervals, baseline comparison included.
- Inventory engine: velocity/survival + bifurcation + repricing + chart endpoints, shadow-mode harness.
- Gap engine: market-driven recommendations with reasons.
- LLM narration: explanations that provably never introduce a number absent from structured input.
- Service API documented (OpenAPI); audit log + monitoring live; secrets externalized; README with setup, env vars, run commands, and the accuracy report.
- Honest limitations (§14) surfaced in outputs and docs.

---

*End of brief. Build for correctness and auditability first; speed and polish second. When in doubt between a convenient shortcut and a verifiably-correct path, take the correct path and document the tradeoff.*
