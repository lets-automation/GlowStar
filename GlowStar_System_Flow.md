# GlowStar AI Pricing — How It Works

The system runs **two processes**. **Process 1** *learns* (once a night).
**Process 2** *prices* (every request). The price is a **50/50 blend of two
independent opinions** — the model (your history) and the live market.

**Legend:**  🟢 live API   🟡 fed / scheduled file   ⚙️ computed

---

## Process 1 — Nightly Training  *(builds the model)*

```mermaid
flowchart TD
    CP["🟢 Channel Partner API<br/>your latest sales"]:::live
    SNAP["Bank daily snapshot<br/>+ union all history"]:::compute
    SOLD["~20,000 sold stones<br/>what you sold and at what discount"]:::data
    FEAT["Build features  (no leakage)<br/>shape · size+bracket · colour · clarity<br/>cut/CPS · fluorescence · lab · Rap · recency"]:::compute
    TRAIN["⚙️ Train the model<br/>gradient-boosted, recency-weighted<br/>learns: stone attributes  =&gt;  the discount YOU set<br/>+ confidence bands + asking-to-realized offset"]:::compute
    GATE{"Accuracy gate<br/>out-of-time test"}:::compute
    MODEL["✅ Live model<br/>versioned"]:::model

    CP --> SNAP --> SOLD --> FEAT --> TRAIN --> GATE
    GATE -->|"promote ONLY if accuracy holds"| MODEL

    classDef live fill:#1e6b3a,stroke:#2ecc71,color:#fff
    classDef compute fill:#26384a,stroke:#5dade2,color:#fff
    classDef data fill:#3a3a3a,stroke:#9aa,color:#fff
    classDef model fill:#3a2c4d,stroke:#a569bd,color:#fff
```

> Training fetches your **sales** (Channel Partner). It does **not** fetch the
> market — that happens at pricing time.

---

## Process 2 — Pricing a Stone  *(runs the model + live market, every request)*

```mermaid
flowchart TD
    STONE["A stone to price<br/>shape · carat · colour · clarity<br/>cut · fluorescence · Rap"]:::data

    subgraph OP1["Opinion 1 — the MODEL  (your history)"]
        M["Predicts a discount from the stone's attributes.<br/>Works even with NO market data.<br/><b>e.g. -57.8%</b>"]:::model
    end

    subgraph OP2["Opinion 2 — the LIVE MARKET"]
        MKT["🟢 Uni API: pull comparables<br/>for this cut + 4C segment"]:::live
        CLEAN["Clean them:<br/>drop duplicate re-listings · trim outliers<br/>separate BGM/milky · asking-to-realized<br/><b>e.g. market -57%</b>"]:::compute
        MKT --> CLEAN
    end

    BLEND["⚙️ BLEND  =  50% model  +  50% market<br/>half what YOU realize (history),<br/>half where the market is now"]:::blend
    ADJ["Adjust + flag<br/>BGM deduction (if recorded) · feedback correction<br/>🟡 Rap-change red-line · confidence band"]:::compute
    OUT["FINAL OUTPUT<br/><b>discount -59.4%</b> · $/ct · net · fair range<br/># market comps · plain-English 'why' · flags"]:::out

    STONE --> M
    STONE --> MKT
    M --> BLEND
    CLEAN --> BLEND
    BLEND --> ADJ --> OUT

    classDef live fill:#1e6b3a,stroke:#2ecc71,color:#fff
    classDef compute fill:#26384a,stroke:#5dade2,color:#fff
    classDef data fill:#3a3a3a,stroke:#9aa,color:#fff
    classDef model fill:#3a2c4d,stroke:#a569bd,color:#fff
    classDef blend fill:#1f5673,stroke:#5dade2,color:#fff
    classDef out fill:#145a32,stroke:#27ae60,color:#fff
```

> If the live market is unavailable it falls back to the banked aggregate — it
> never invents data. For a stone with **no** comparables, the **model alone**
> sets the price (that is what the model is for).

---

## Where every number comes from

| Used at | Source | Live / static |
|---|---|---|
| Training — your sales | Channel Partner API | 🟢 live, nightly |
| Pricing — market comps | Uni API | 🟢 live, per stone |
| Pricing — the model | versioned trained model | 🟢 refreshed nightly |
| Rap (your stones) | comes in your records | 🟢 live |
| Rap (new stones, no price) | versioned Rapaport list | 🟡 fed weekly* |
| Fallback market (if Uni down) | banked Uni aggregate | 🟡 periodic rebuild |

\* No live Rapaport API exists for us; the list is fed weekly (or via RapNet
when provisioned). The **Rap-change red-line fires automatically** once a new
list is loaded.

---

## In one sentence

**(In production)Every night the system re-learns your pricing from your latest sales; every
time it prices a stone it runs that model *and* pulls the live market, then
blends them, so the price reflects what *you* would realize, kept current by
where the market is now, and ships with a range, a reason, and honest flags.**
