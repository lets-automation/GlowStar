# GlowStar — AI diamond-pricing engine

Client: **Glow Star** (natural diamonds, India). The engine prices a stone as a
**discount off Rapaport** (`FDiscount`, negative = deeper). The client's desk
returns our files with their own prices and grades us on the variance, so being
*explainably* right matters as much as being right.

> **This file holds invariants and traps only — never numbers.**
> Model version, MAE, row counts, test counts and open-item status rot within days.
> An earlier brief hardcoded them, they went stale, and a fresh session trusted the
> stale version. For anything current, RUN IT:
>
> ```
> python -m glowstar.status      # model, MAE, data freshness, fluoro caps, feedback state
> python -m pytest -q            # test count / health
> ```

---

## Architecture (Engine 1 — built, in use)

**One config ships: `training/retrain.serving_config()`.** Training, the gate and
serving all construct it, and `_assert_gate_scores_what_ships` fails the retrain if
they diverge. This is load-bearing — see Trap 5.

- **GBM** (HistGradientBoosting, quantile) on the client's realized sales,
  recency-weighted, leakage-guarded (`features/build.py` whitelist).
- **Grid-routed second model** (`_fit_grid_model`) — same features **plus** the
  client's own Master-grid reading for the stone's cell, joined **point-in-time**
  (`market/grid_history.py`). Fit only on rows that have a cell; used only for
  stones that have one. Everything else uses the plain model. Do not merge them:
  routing wins at both ends and removes the dead-feed cliff.
  *Why it exists:* the dominant error is the **current per-cell level**. Correcting
  it (oracle) halves MAE and cuts the ≥5pt tail ~4x, while global/weekly corrections
  do almost nothing. A tree cannot supply it — it cannot extrapolate time, and ~40%
  of served stones fall past its training window.
- **Market anchor: OFF (`anchor_lambda=0.0`)**. Measured monotonically harmful — it
  is an *asking* level ~6 pts shallower than where this desk sells, and blending it
  drags every price toward "too expensive to trade". The grid model does the
  level job properly. `market_led=True` is worse still and is ablation-only.
- **Competence guard** — shapes where the model loses to a segment-median baseline
  are deferred to it, measured at fit time (not a hardcoded list).
- **Registry + promotion gate** — a candidate must beat the incumbent within
  tolerance to go live. **The gate works; it has already caught a bad model. Never
  bypass it.**

Rounds use the client's irregular size slots (`market/segments.py`).

### The grid: never an anchor, always a feature
"Never copy the grid" is **correct and verified point-in-time** — do not undo it.
With the grid joined as of the day *before* each sale, the grid **alone** loses to
the engine badly and the best engine/grid **blend weight is 0%**. Using today's grid
to explain a past sale reverses that result and makes blending look brilliant; that
is leakage, and it is a trap that has caught me. The grid's value is as a *feature*
the model can learn to trust or discount — never as the answer.

---

## Traps — read before touching these

### 1. Fluorescence — I got this wrong and the client caught it
The GBM's fluoro penalty is **CORRECT for D–E at Medium/Strong/V.Strong** (deep).
Fluorescence genuinely guts a colourless stone, and the desk really does quote it
that deep. The GBM **only over-penalises where fluoro barely matters**: low colours
(**I–M**) and the **Faint** tier.

`_compute_fluor_caps()` therefore caps **only I–M and Faint. Never cap D–E at
Medium+.** Doing so flattens a real effect and puts stones 5–10 pts off the desk's
price.

**Root cause of my error:** I derived caps from *pooled realized-sale medians*.
Those are **confounded** (different size/clarity mix per fluoro tier) and are **not
the desk's pricing rule**. Validate against the desk's **actual quoted prices**.

### 2. The override echo — never hand the desk their own prices back
`set_feedback()` builds `{stone_id: the desk's exact price}`. If that is applied
when re-pricing a batch the desk already corrected, **the file replays their own
numbers at them** — every flagged stone lands on target to the decimal. It looks
like a triumph and measures **nothing**; it would collapse on the next file and
destroy the credibility this project runs on.

`price_and_report(use_feedback=False)` is the **default**. Prices are model+market
only.

**Red flag:** variance that looks *too good*, or many stones at exactly 0.0 off.
Always ask *"could this number be leaking the answer?"* before celebrating.

### 3. Feedback is recorded but DISABLED — do not just switch it on
The desk's returned price (`glow price` column — **negative sign**; variance =
`our_sale - abs(glow)`) is an **asking quote, not a realized sale**. Training a
sale-price model on quote labels teaches the wrong target.

Measured: enabling feedback **costs ~+0.9 MAE** (segment corrections and training
labels both hurt). `build_corrections(min_support=3)` shifts a whole price cell off
3 stones.

- Training: `GS_USE_FEEDBACK=0` (default). **Left on, the gate rejects every
  candidate and the nightly retrain silently freezes.**
- To use it properly: raise `min_support` (~8–10), shrink the offsets, and score
  **both** objectives — realized-sale MAE **and** variance-vs-desk-quote. Tune on
  held-out data; the returned batch is small and easy to overfit.

### 4. Tinge is a model feature, not a deduction
The client supplies tinge live as **structured fields**: `Brown` / `Milky` / `Shade`
/ `Green` (codes like `LBR`, `MML`, `HMT`; ~99% populated). The loader parses them to
`brown_ord` / `milky_ord` / `shade_ord` / `green_ord` and feeds the **model**. Do
**not** also apply a post-model deduction — that double-counts.

- The legacy free-text `BgmComments` carried **only brown and milky** — shade (tint)
  and green were invisible, and both are genuinely priced. It is now a backfill for
  old snapshots only; prefer the structured fields.
- "Unassessed" is the sentinel `UNASSESSED` (-1.0), **never NaN and never 0.0**.
  0.0 means *assessed and clean* — conflating them silently over-prices a tinged
  stone. And an all-NaN column hard-crashes HistGradientBoosting at fit, so a feed
  that stops sending a field would take the nightly retrain down.
- Known limit: the model can't grade *severity* well (few Heavy examples), so
  Medium/Heavy tinge is **flagged for the desk to review** rather than silently
  under-priced.

### 5. Measure the path that SHIPS, not the one that's convenient
`price_and_report` shipped `market_led=True` for weeks while the gate, `status` and
every backtest scored a bare `EngineConfig()` (`market_led=False`). **The published
accuracy was from a pipeline the client never received** — on the same stones the
shipped path was roughly *twice* the error and its "80% band" held ~56% of the time.
Nothing was hardcoded and no test failed; the two paths simply drifted apart.

Hence `serving_config()` + `_assert_gate_scores_what_ships()`. If you add a config
knob that changes a price, it goes in `serving_config` or it does not ship.
**Before trusting any A/B, reconstruct the shipped number first.**

This bug has THREE heads, and all three were live at once — anything that evaluates
the engine must evaluate the **shipped** pipeline (`_fold_predict`), not a
convenient stand-in:
- the **gate** scored `market_led=False` while the client got `market_led=True`;
- the **conformal** calibrated a bare model with no grid routing, so the stated 80%
  band was a promise about a function we don't ship;
- the **competence guard** benched shapes against a bare non-grid model, then
  deferred the ones it lost on to a median baseline. Measured against the desk's
  own quotes those deferred shapes scored MAE 3.87 / **38.5% >=5pts out**, versus
  1.54 / 6.4% for model-priced stones. Fixing the guard to judge the shipped
  pipeline cut the file's >=5pt tail from 9.8% to 5.7% — **it was the tail**.

If you find a fourth, assume the same shape: something is measuring a model that
never reaches a client.

### 6. The client's data is deterministic — so "it's just noise" is a lie
Identical stones (same spec, same month) sell within **~0.4 pts**; an oracle on the
group median is ~0.5 MAE. So when the desk says "you are not wrong by this margin,"
**they are right** — a wide tail is model error, not negotiation noise. I nearly
reported the tail as irreducible. Don't. Find the cause.

Corollary: features are not automatically the cause either. The client's own
measurements (table/depth/ratio/angles) and their internal make grade `MGrade`
were measured to give **no significant gain** — the error was never about the stone.
Test before you believe.

### 7. The DISCOUNT is not monotone in colour/clarity — do not "fix" it
Sweeping colour D→M shows ~40% of adjacent pairs "inverted" (a worse colour priced
shallower). It looks like an obvious bug, and a monotonic constraint drives it to
~1% for ~0.02 MAE — free virtue. **It is wrong, and I nearly shipped it.**

The target is a discount **off Rap**, and Rap already prices colour and clarity, so
the discount is the *deviation*. Measured on the client's own realized sales,
**47.7%** of well-supported adjacent colour pairs have the worse colour at a
shallower discount — F/G/H are commercial goods and trade shallower off Rap than
hard-to-move D/E. The "inversions" are the market. PRICE ($/ct) is far more monotone
(~18% reversals) — that is the invariant the desk judges. Constrain price, or
nothing. Guarded by `test_discount_is_not_constrained_monotone`.

### 8. Grid freshness is the whole ballgame
Measured by cell age at sale: fresh cell (0–3d) → MAE **1.99**; 30d+ stale → **3.10**.
The grid model is only as good as the daily refresh. If the snapshot job stops, the
grid feature quietly rots and takes the price with it. `grid_age_days` is a model
feature and the report shows cell age — keep both.

---

## Data rules

- **LIVE for everything.** Never price or train from static/sample files. A static
  sample once made me tell the client something was impossible when it wasn't —
  the live API disproved it.
- `records.json` is rebuilt fresh from the live API on every retrain.
- **Rapaport list is STATIC** (April CSV; no live feed exists). Affects `$/ct`, not
  the discount.
- Credentials live in `.env`. `ingestion/http.py` sanitises URLs in errors — some
  creds are path-embedded and used to leak into job logs.

- **Grid history is data you cannot backfill later.** `market/grid_history.py`
  accumulates the point-in-time grid; the daily job appends to it. Without it the
  grid model cannot be trained OR honestly validated. Never delete it.
- Grid ingestion keeps **all sheets** and canonicalises shape on **both** sides
  (`canon_shape`). Both were real bugs: filtering to the "Master" sheet dropped
  ~99% of CUSHION / 81% of PEAR cells, and the grid spells oval **`F.OVAL`** *and*
  `OVAL`, so ovals silently missed their cell. A missed cell is not cosmetic — the
  stone falls to the interpolated grid estimate, which is ~5x worse than a real
  cell and manufactured the client's "you're 20 points off your own grid" escalation.
  **Never show an interpolated grid number to the client.**

Scheduled (Windows Task Scheduler): snapshot + retrain nightly. Caveat: it's a
local machine, so they only run when the box is on. The tasks were also silently
dying (`0xC000013A`) on battery/idle settings — if the logs go quiet, check the
task settings before you trust the model in the registry.

---

## Working rules

- **Accuracy over agreement.** The user has been burned by premature "it's fixed"
  claims — including by me. Say what you measured, not what you hope.
- **Measure before shipping.** Reconstruct-and-assert: prove your counterfactual
  reproduces the **shipped** number before trusting any A/B. A sign bug once
  inverted an entire policy table and nearly shipped the wrong fix.
- **Verify, don't assume — including about other agents' code.** I nearly reported
  a colleague's correct guard as a bug; checking the types took one command.
- Deliverables go to `artifacts/`. Never ship a file you haven't scored against the
  desk's own numbers.
