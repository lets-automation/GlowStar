# End-to-End Testing Guide — Glow Star Pricing Engine

How to verify every part of the engine, from a single function to the full
pipeline, plus the acceptance gates that decide go-live. All commands assume the
venv is active (`. .venv/Scripts/activate`) and you are in the repo root.

Nothing here is a stub: every command runs against the real shipped data
(`records.json`, the Rap CSVs, the 6.2 GB Uni dump's artifacts). Live-API steps
run the moment credentials are in `.env`; until then they fall back to the
banked data so the pipeline still runs end to end.

---

## 0. One-command confidence check

```bash
pytest -q                                   # 93 tests, ~40s — must be all green
python -m glowstar.validation.engine_backtest   # accuracy gate (numbers below)
python -m glowstar.validation.shadow             # go-live recommendation
python -m glowstar.pipeline                       # full ingest->train->price
```
If those four succeed, the engine is working end to end. The rest of this doc
explains what each proves and how to dig in.

---

## 1. Unit & integration tests (`pytest`)

93 tests across 10 files. Run all, or a slice:

```bash
pytest -q                                   # everything
pytest tests/test_rap_lookup.py -q          # deterministic Rap core (25 tests)
pytest -k leakage -q                        # the leakage guard
pytest tests/test_engine.py -q              # full engine behaviour
```

What the critical ones prove:

| Test file | Guarantees |
|---|---|
| `test_rap_lookup.py` | Rap $/ct is exact vs known cells; 6–9.99ct gap, oversize, undersize, fancy color, FL→IF all return explicit status — never a silent wrong number. |
| `test_loaders.py` | 28,408 records load; the price identity holds (≤2¢); 19 outliers flagged not dropped; 6-month range. |
| `test_features_model.py` | **No forbidden/transaction column can reach the model** (leakage guard raises); intervals stay ordered. |
| `test_engine.py` | Suggestions well-formed; rare/fancy route to fallback; **MAE stays < 6** (regression guard); `bgm_unassessed` flag set; online feedback correction shifts price. |
| `test_market.py` / `test_authenticity.py` | Uni codebook is confirmed-only (fails loud otherwise); dedup/median/source-quality cleaning. |
| `test_bgm.py` | BGM-as-base: unassessed/clean/bgm states + deductions. |
| `test_feedback.py` | Rejections require a reason; overrides require a price; corrections + retrain labels + analytics. |
| `test_market_trend.py` | Trend index + drift/projection; macro provenance; internal-vs-macro cross-check. |
| `test_ingestion.py` | Credentials env-only; HTTPS enforced; secrets never logged; 4 connectors (mocked); immutable snapshots + schema drift. |

---

## 2. Accuracy gate — the out-of-time backtest

The credibility core. Trains on sales **before 2026-05-01**, tests on **May–Jun**,
leakage-free, and compares to a transparent baseline.

```bash
python -m glowstar.validation.engine_backtest
```

Expected (current build):

```
ENGINE    MAE=4.99  MedAE=4.58  ±3=32.5%  ±5=54.7%  $MedAE=48  $MAE=119
BASELINE  MAE=7.36  MedAE=6.50  ±3=28.0%  ±5=42.3%  $MedAE=66  $MAE=178
-> engine beats baseline by 2.36 MAE pts (32.1% lower); signed bias +0.25
CONFIDENCE target=80% empirical=64.2%
```

**How to read it / acceptance gates (agree final numbers with the client per
segment — do not hardcode one global pass mark):**
- **MAE in discount points** — lower is better; must clearly beat the baseline.
- **±5 share** — fraction priced within 5 discount points of the realized sale.
- **signed bias** — near zero means no systematic over/under-pricing.
- **$ errors** — business impact per stone.
- **coverage** — the 80% band should contain truth ~80% of the time. It is
  currently **64%** (honest): 6 months out-of-time across a market regime shift
  means the calibration window under-represents test dispersion. It rises as the
  daily snapshot job banks data. Point accuracy is unaffected.

Vary the split to stress different windows:
```bash
GS_BACKTEST_SPLIT=2026-04-01 python -m glowstar.validation.engine_backtest
```

### Prove there is no leakage yourself
```bash
python -c "from glowstar.features.build import build_features; from glowstar.data.loaders import load_records, sold_stones, FORBIDDEN_FEATURES; df,_=load_records(); x=build_features(sold_stones(df)); print('forbidden cols present:', FORBIDDEN_FEATURES & set(x.columns))"
# -> set()  (empty: FDiscount/Discount/NetAmount/etc. never enter the model)
```

---

## 3. Shadow mode — go-live gate

Runs the engine silently against the human pricers' decisions and recommends
go-live only where agreement is high.

```bash
python -m glowstar.validation.shadow
```

Reads as: engine agrees with the human within ±5 on ~55% of stones overall
(mean gap +0.2 — no bias); **Round is recommended for go-live (61% agreement)**;
other shapes hold (engine assists, human decides); ~7% material divergences
become the human-review queue. Note (documented in the harness): on sold stones
the human's listed `Discount` ≈ realized `FDiscount`, so this measures
*agreement*, not a rigged "beat the human" contest.

---

## 4. Market data: build artifacts + authenticity report

The market anchor + BGM deductions come from the Uni dump. Build/refresh:

```bash
python -m glowstar.market.aggregate_bulk        # streams 6.2GB (~2 min)
# -> artifacts/market_segments.json, artifacts/bgm_discounts.json
cat artifacts/bgm_discounts.json                 # milky/shade discount ladder + dedup counts
```

Authenticity is verifiable in the output: `counts.duplicates` shows ~90% of the
raw feed were re-listings (deduped by certificate); the milky ladder
(−5/−10/−11) is monotonic. Test the cleaning pipeline on any live pull:

```bash
python -c "from glowstar.market.authenticity import clean_market_stones; import json; raw=json.load(open('response-uni.json'))['data']; r=clean_market_stones(raw); print(r.report.as_dict())"
```

---

## 5. BGM-as-base (client request)

Verify the three states from one command:

```bash
python -c "
from glowstar.service.pricing_service import PricingService, StoneIn
svc=PricingService()
for p in [dict(StoneId='u',Shape_full='Round',Weight=1.01,Color='G',Clarity='SI1',CPS='3EX',Lab='GIA',Rap=6500),
          dict(StoneId='c',Shape_full='Round',Weight=1.01,Color='G',Clarity='SI1',CPS='3EX',Lab='GIA',Rap=6500,milky='No Milky',Shade='None'),
          dict(StoneId='b',Shape_full='Round',Weight=1.01,Color='G',Clarity='SI1',CPS='3EX',Lab='GIA',Rap=6500,milky='Medium Milky')]:
    s=svc.price(StoneIn(**p))['suggestion']
    print(s['bgm_state'], s['suggested_discount'], 'deduction', s['bgm_deduction_pts'], 'flags', s['flags'])
"
# unassessed -57.1 (flag bgm_unassessed) | clean -57.1 | bgm -67.1 (deduction -10)
```

The base price is always the **No-BGM clean** stone; BGM is an explicit
deduction; unknown BGM is flagged so you never pay clean price for a BGM stone.

---

## 6. Human feedback loop (client request)

Record a pricer's decision and watch the model learn immediately, then verify
durable learning on the next retrain.

```bash
python -c "
from glowstar.service.pricing_service import PricingService, StoneIn
svc=PricingService(use_feedback=False)
st=StoneIn(StoneId='F1',Shape_full='Round',Weight=1.01,Color='G',Clarity='SI1',CPS='3EX',Lab='GIA',Rap=6500)
s=svc.price(st)['suggestion']; print('before:', s['suggested_discount'])
for _ in range(4):
    svc.record_decision(stone=st, decision='override', suggested_discount=s['suggested_discount'],
        suggested_net=s['suggested_net'], reason_code='discount_too_deep',
        human_discount=s['suggested_discount']+6, note='buyer paid more', user='pricer1')
print('after 4 overrides:', svc.price(st)['suggestion']['suggested_discount'])   # shifted +6
"
```

- **Online (immediate):** per-segment median override shifts future suggestions
  in that segment at once.
- **Durable (next retrain):** `PricingService()` loads `data/feedback/decisions.jsonl`
  and folds OVERRIDE (gold, up-weighted) + ACCEPT labels into `engine.fit`.
- **Analytics:** `record_decision` returns `feedback_summary` (acceptance rate,
  reason counts) — drives what to fix next (e.g. many `bgm_present` → push CRM
  BGM capture).

Rejections **must** carry a reason; overrides **must** carry the human's price
(validated — see `test_feedback.py`).

---

## 7. Full pipeline (ingest → train → serve)

```bash
python -m glowstar.pipeline
```
Tries a live Channel Partner pull; with no credentials it falls back to the
latest banked snapshot, else the shipped `records.json` — then trains on all
sold history + feedback and prices a sample stone with full market context. This
is the "everything works end to end" check.

---

## 8. Service / REST API

```bash
# In-process:
python -c "from glowstar.service.pricing_service import PricingService, StoneIn; import json; print(json.dumps(PricingService().price(StoneIn(Shape_full='Round',Weight=1.0,Color='G',Clarity='VS2',Lab='GIA',Rap=8000)), default=str, indent=2))"

# REST (optional dep): pip install fastapi uvicorn
uvicorn glowstar.service.app:app --reload
#   GET  /health           -> {"status":"ok"}
#   POST /price  {stone}    -> suggestion + interval + market context + explanation
#   OpenAPI docs at /docs
```

The LLM explanation is guarded: set `ANTHROPIC_API_KEY` to use Claude
(`GS_LLM_MODEL`, default `claude-opus-4-8`); without it a deterministic template
is used. Either way the **number guard** rejects any figure not in the computed
facts (`tests/test_narration.py`).

---

## 9. Live data — CONNECTED and working

Credentials are in `.env` and all four APIs return live data. Verify and use:

```bash
# Smoke-check all four live endpoints:
python -c "import glowstar.config; from glowstar.ingestion import channel_partner, diamanto; print('records', len(channel_partner.get_all_records())); print('token ok', bool(diamanto.get_access_token()))"

# Verify/refresh the Uni codebook against the LIVE API (writes artifacts/uni_codebook.json):
python -m glowstar.market.calibrate_codebook

# Pull live and bank an immutable snapshot:
python -m glowstar.ingestion.run_snapshot
```

**The LIVE Excel report** — live records + live Uni comparables per stone, real
actual-vs-suggested, no hardcoded values:
```bash
python -m glowstar.reporting.live_report     # -> artifacts/GlowStar_Pricing_Report_LIVE.xlsx
```
(Takes ~15-20 min: the Uni export-report endpoint is ~10-15s per segment; calls
are cached per segment. Smaller sample: edit `build(n=...)`.)

Schedule the snapshot job daily (Task Scheduler / cron lines in
`run_snapshot.py`) to bank history from day one. Rotate credentials before wider
production rollout.

Note: live calibration **corrected a documentation error** — the API doc implied
Uni clarity code 1 = IF, but the live API returns 1 = FL, 2 = IF. Always trust
the calibrated codebook over the doc.

---

## 10. Acceptance checklist (go-live)

- [ ] `pytest` all green.
- [ ] Backtest beats baseline on agreed per-segment thresholds; intervals
      reported with honest coverage.
- [ ] No leakage (forbidden-cols check returns empty).
- [ ] Shadow mode: agreed segments meet the agreement bar (start: Round).
- [ ] Market artifacts fresh; authenticity report reviewed (dedup rate, ladder).
- [ ] Snapshot job scheduled and producing daily files.
- [ ] Feedback loop writing `decisions.jsonl`; corrections applying.
- [ ] Secrets externalised; live connectors authenticate; codebook confirmed.

Suggested CI: run `pytest -q` and `engine_backtest` on every change; fail the
build if engine MAE regresses past an agreed threshold (the `test_engine.py`
guardrail already enforces MAE < 6).
