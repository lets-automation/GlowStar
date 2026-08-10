# Pricing Engine — Integration Guide

**For:** Glow Star IT (Jay Bhai)
**From:** Lets Automation
**Covers:** what your software sends us, and what it gets back.

---

## 0. The service is LIVE — connection details

The server is built and the engine is running. You can start integrating today.

| | |
|---|---|
| **Base URL (testing)** | `http://217.217.248.111:8000` |
| **Base URL (go-live)** | An `https://…` address — **will change, see below** |
| **Authentication** | Header `X-API-Key: <the key we send you separately>` |
| **Format** | JSON in, JSON out (`Content-Type: application/json`) |
| **Rate limit** | 120 requests/minute. Over that returns `429` with a `Retry-After` header. Far above normal desk use — it only triggers on a runaway loop. |
| **Max batch** | 5,000 stones per call (larger returns `413`) |

> ⚠️ **Please put the base URL in a configuration setting, not hardcoded.**
> The address above is plain HTTP for integration testing. Before your desk uses
> this for real prices we will move it to a proper `https://` address, and the
> only thing that should need to change on your side is one config value.

**Quick test** — this should return a price:

```bash
curl -X POST http://217.217.248.111:8000/price \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"Shape_full":"ROUND","Weight":1.01,"Color":"G","Clarity":"VS1"}'
```

### Codes: send us whatever your system already uses

You do **not** need to translate anything. We accept your trade codes and
normalise them at our end:

| Field | All of these work |
|---|---|
| Shape | `RBC`, `ROUND`, `Round`, `round`, `BR` → Round. Likewise `OB`/`OVAL`, `PB`/`PEAR`, `MB`/`MARQUISE`, `HB`/`HEART`, `EM`/`EMERALD`, `CCRMB`/`RADIANT` |
| Cut/Polish/Symmetry | `3EX`, `EX-EX-EX`, `EX EX EX`, `EX/EX/EX` all mean the same thing |
| Fluorescence | `NON`/`None`, `FNT`/`Faint`, `MED`/`Medium`, `STG`/`Strong`, `VSTG`/`Very Strong` |

Send whichever form your inventory already holds. Sending an unknown shape is
safe — the stone is flagged for review rather than silently mispriced.

---

## 1. What this replaces

Today we email an Excel file, your desk marks it up, and emails it back.

After integration, your own screen shows the price. Your desk accepts or changes
it there, and that decision comes straight back to us. No more files.

Two connections, that is all:

| # | Direction | Purpose |
|---|---|---|
| 1 | You ask us | "What should this stone sell for?" |
| 2 | You tell us | "Here is what the desk decided." |

---

## 2. Connection 1 — asking for a price

**Send to:** `POST https://<server>/price`
**Header:** `X-API-Key: <the key we give you>`

### What you send

Only four fields are required:

```json
{
  "Shape_full": "Round",
  "Weight": 0.52,
  "Color": "G",
  "Clarity": "VS1"
}
```

**Please also send these whenever you have them.** Each one makes the price more
accurate, and they are all fields your inventory already holds:

```json
{
  "StoneId": "S26-10545",
  "Shape_full": "Round",
  "Weight": 0.52,
  "Color": "G",
  "Clarity": "VS1",

  "CPS": "3EX",
  "Fluorescence": "Non",
  "Lab": "GIA",
  "Location": "IND",

  "Brown": "NO",
  "Milky": "LML",
  "Shade": "NO",
  "Green": "NO",

  "Length": 5.18,
  "Width": 5.21,
  "Depth": 3.19
}
```

**Notes**

- **You do NOT need to send the Rapaport price.** We hold the licensed sheet and
  look it up ourselves. This is deliberate: Rapaport re-bases one size band at a
  time, and if two systems hold two copies of the sheet, one of them is eventually
  wrong and nothing on screen looks unusual. One sheet, one source. (If you ever
  *do* want to force a specific Rap — to reproduce an old quote — send `Rap` and
  we will use yours.)
- **Send the tinge codes exactly as your API already emits them** — `LBR`, `MML`,
  `HMT`, `NO`. No conversion needed at your end.
- **Length and Width** let us apply the face-up size (spread) premium on rounds.
- Anything you omit is handled; the price simply carries less information.

### What you get back

```json
{
  "suggestion": {
    "stone_id": "S26-10545",
    "suggested_discount": -43.78,
    "suggested_ppc": 1068.14,
    "suggested_net": 555.43,
    "ci_discount_low": -47.30,
    "ci_discount_high": -38.97,
    "comparable_count": 1763,
    "method": "model+anchor",
    "flags": []
  },
  "explanation": { "...": "plain-English reason for the desk" }
}
```

| Field | Meaning |
|---|---|
| `suggested_discount` | **The answer.** Percent below Rapaport (negative). |
| `suggested_ppc` / `suggested_net` | The same price in $/ct and total $. |
| `ci_discount_low` / `_high` | The fair range. We expect the true price inside this 80% of the time. |
| `comparable_count` | How many similar market stones this is based on. |
| `flags` | Any warning. **`fluor_review` or `bgm_review` means: please have the desk set this one.** |
| `explanation` | Why we said this, in words your desk can read. |

**Many stones at once:** `POST /price/batch` with a list (max 5,000). One bad
stone never fails the rest — failures come back individually with their reason.

---

## 3. Connection 2 — sending the decision back

**Send to:** `POST https://<server>/decision`

This is the half that makes the engine improve. Every accept and every correction
is stored permanently.

```json
{
  "stone_id": "S26-10545",
  "decision": "override",
  "suggested_discount": -43.78,
  "human_discount": -45.00,
  "reason_code": null,
  "user": "milan"
}
```

- `decision` is `accept`, `reject`, or `override`.
- `human_discount` is the desk's own discount. **If your screen works in dollars,
  send `human_ppc` instead and we convert it** — your desk never does arithmetic.

### The 2-point rule (as your desk asked)

| Difference between our price and the desk's | Reason needed? |
|---|---|
| **2 points or less** | **No.** Just the price. |
| **More than 2 points** | **Yes** — a reason code is required. |

A small difference is ordinary trading judgement. Forcing a reason there only
teaches people to pick any value to clear the form, which makes the reasons
useless. A large difference is worth explaining, because that is what teaches the
engine *why* it was wrong instead of only *that* it was.

The reply tells you which case you are in, so your screen can show the same rule:

```json
{ "recorded": true, "variance_pts": 1.22, "threshold_pts": 2.0,
  "needs_attention": false, "reason_required": false }
```

If a reason is required and missing, you get a clear `422` with a message you can
show the user. Nothing is silently dropped.

**Reason codes:** `discount_too_deep`, `discount_too_shallow`, `bgm_present`,
`make_quality`, `market_moved`, `special_situation`, `data_error`, `rare_item`,
`other`.

---

## 4. Health check

`GET /health` — no key needed, safe for a monitor to poll.

```json
{ "status": "ok",
  "model": { "version": "20260728T102722", "test_mae": 2.469 },
  "records_age_hours": 24.1 }
```

`status` becomes `degraded` if the model cannot load or the data is over 48 hours
old. **Please alert on that** — it is the early warning that something upstream
stopped.

---

## 5. What we need from you

### 5.1 A server — DONE

The server is built, secured and running (Ubuntu 24.04, Mumbai, always on). The
price service, the nightly retrain and the daily data pulls all run on it. There
is no separate "model server". **Nothing further is needed from you here.**

**Why always on matters — measured, not theoretical.** The engine refreshes your
price grid every day. When that refresh is current our error is about **2
points**; when it is two weeks behind, the same engine is about **3 points** out.
The old setup ran on a laptop that slept and missed 10 nights in 23. The new
machine simply never stops, and if it is ever restarted the missed job runs
automatically as soon as it comes back.

### 5.2 Confirmations

1. Your CRM can call an external HTTPS endpoint (any firewall rule needed?).
2. The tinge fields (`Brown`/`Milky`/`Shade`/`Green`) will keep coming through as
   they do today.
3. Who receives the alert if `/health` reports `degraded`?

**Nothing else.** Inventory, price grid and market data are already connected and
running.

---

## 6. How the engine improves over time

**Two different things, deliberately kept apart.**

**A. Your sales — teaching. Already running.**
Every night the engine studies every stone you actually sold and rebuilds itself.
This is live today and is where accuracy comes from.

A new version only replaces the current one **if it scores better on stones it has
never seen**. If a night's data is bad, the previous version keeps working. This
is why a bad day cannot reach your desk.

**B. Your corrections — grading. Stored, not yet taught.**
When your desk changes our price, that is usually the price you are *asking*,
which is not the same number as the price a stone eventually *sells* for. We
tested training on those directly: accuracy got **worse**. So for now corrections
are stored and used to measure us, not to teach the engine.

We do not have to remember to revisit this. The nightly job prints exactly where
it stands:

```
feedback readiness:
  decisions 124 | priced overrides 55/150 | supported cells 0/25
  desk moves us -2.30 pts (median); 69% of the time deeper
  still needed: 95 more priced overrides; 25 more price cells with 8+ corrections
  -> Keep collecting.
```

When the volume is sufficient, we run a controlled comparison and switch it on
**only if it improves both** real-sale accuracy and agreement with your desk. You
will see the evidence before anything changes.

---

## 7. Suggested rollout

| Step | What happens |
|---|---|
| 1 | You provide the server; we deploy and share the API key |
| 2 | You connect a test screen; we check prices against a batch you know |
| 3 | Desk uses it live alongside the current process for ~2 weeks |
| 4 | Excel retires |

Steps 1–2 take a few days once the server exists.

---

## 8. Questions

Ask us. If any field or reply here is unclear, tell us and we will change it — it
is easier to fix the contract now than after your screens are built.

---

# Appendix A — Response to "For FrontOffice API" (29-07-2026)

We have built all three endpoints to your specification. Field names and response
shape are exactly as your document asks, so your CRM can bind to them directly.

| Your spec | Endpoint | Status |
|---|---|---|
| 1. Bulk stone pricing | `POST /frontoffice/price` | **Ready** |
| 2. Add our reason | `POST /frontoffice/reason` | **Ready** — see A.2 |
| 3. Pricing Master Discount AI | `POST /frontoffice/master-discount` | **Ready** |

## A.1 — Bulk pricing: what each response field is

Per stone you get: `StoneId`, `CertificateNo`, `AIDiscount`, `AIPricePerCarat`,
`AITotal`, `FairRangeLow/High`, `Reason`, `Tradeability`, `TradeabilityDays`,
`TradeabilityBasis`, `ConfidenceScore`, `AIScore`, `NeedsReview`, `Flags`.

**Three things to know before you build the screen.**

**(a) We accept every field you send — but we only PRICE on the ones our sales
history contains.** Your request includes `eyeClean`, `luster`, `bowtie`,
`iGrade`, the black/white/open/natural family, `kapan`, `article`, `grade`. We
checked your live inventory API: those fields are not in it, so they are not in
the sales history either, so the model has never seen them and cannot price on
them. We store them and list them back in `ReceivedNotYetPriced` — you will see
exactly what did and did not influence the number, rather than assuming.

As those fields start appearing in the daily data they become learnable, and we
add them deliberately with a measured before/after.

**Used today:** shape, weight, colour, clarity, cut/polish/symmetry (or cps),
fluorescence, lab, brown, milky, shade, green, and the measurement block
(length, width, depth, ratio, table, girdle, crown/pavilion angles, mGrade).

**(b) `AIScore` returns `null` for now.** The overall score needs the Demand,
Competition and Liquidity inputs (the eight-parameter score). Demand needs your
search/enquiry data — we tested `GetCustomerSearchHistories` and it works, but it
carries no buyer identity and no offer/video-view events, so per-stone demand is
not yet measurable. **A wrong score is worse than an honest blank**, so the field
is present and null with a status string, not invented.

**(c) `Tradeability` is provisional, and it is honest about why.** It returns your
five buckets (High / Semi High / Medium / Semi Slow / Slow) plus the number of
days and the basis it used.

Two corrections went into this, pulling in opposite directions:

- Counting only **sold** stones looks too fast — the slowest goods have not sold
  yet, so they never enter the average. Stock is therefore counted as "has taken
  at least N days so far".
- But counting **all** stock looks too slow: 17.6% of your stock predates our
  earliest sale record, and the stones that entered *and sold* in that same old
  period are not in the records at all. Keeping the survivors without their
  successes inflates the number.

So we use only stones that entered stock inside the window our records cover.
On your book: sold-only says 42 days, all-stock says 75, **the correct figure is
46**. The full velocity engine (Workstream B) goes further — intervals,
own-vs-market separation, GMROI.

## A.2 — One gap in spec #2, and it matters

Your document sends: `certificateNo`, `reason`, `aiDiscount`.

**It does not include the desk's own price.** Without it we record *that* we were
wrong but never *what right looks like* — there is no number to learn from, so
the reason can only ever be analytics.

Please add **`deskDiscount`** (your discount off Rap, e.g. `-48.0`) — or
`deskPpc` with the stone's Rap. The endpoint accepts it today and the response
tells you which you got:

```json
{ "recorded": true, "trainable": true, "variance_pts": 4.22,
  "threshold_pts": 2.0, "needs_attention": true }
```

`"trainable": false` means we stored the reason but the model learned nothing.

## A.3 — Master Discount AI

Prices a **cell**, not a stone: you send a weight range, we answer at its midpoint
and return the market support behind it, so a thinly-supported cell never reads
like a well-supported one.

## A.4 — What we still need from you

1. **The server** (section 5.1) — the one real blocker.
2. **`deskDiscount` added to spec #2** (A.2 above).
3. If you want a real `AIScore`: enquiry/offer/video-view events, with which
   customer, per stone.
