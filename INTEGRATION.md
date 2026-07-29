# Pricing Engine — Integration Guide

**For:** Glow Star IT (Jay Bhai)
**From:** Lets Automation
**Covers:** what your software sends us, what it gets back, what server is needed.

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

### 5.1 A server

One Linux virtual machine that is **always on**.

| | |
|---|---|
| Where | Azure, same region as your inventory API (fastest, simplest) |
| Size | 4 vCPU, 16 GB RAM, 256 GB disk |
| OS | Ubuntu 22.04 LTS |
| Cost | roughly $70–110 / month |
| Access | SSH for our team |
| Open ports | 443 inbound (your CRM → us) |

**Why always on matters — measured, not theoretical.** The engine refreshes your
price grid every day. When that refresh is current, our error is **1.5 points**.
When it is two weeks behind, the same engine is **4.3 points** out. Today these
jobs run on a laptop that sleeps. Nothing improves accuracy more than a machine
that simply never stops.

The same machine runs everything — the price service, the nightly retrain, and
the daily data pulls. There is no separate "model server".

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
