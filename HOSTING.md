# Where to Host the Pricing Engine — Azure vs Contabo, Linux vs Windows

---

## The short answer

**Hosting does not change the prices we produce.** The same code, the same model
and the same data produce the same numbers on any of these machines. So the
question "will we get the same results cheaper?" has a straightforward answer for
accuracy: **yes — the numbers are identical.**

What changes between providers is **three things**, and only one of them can
touch accuracy:

| What changes | Does it affect the *prices*? |
|---|---|
| How fast one price comes back (1–3 sec vs 2–4 sec) | **No.** Same number, slightly later. |
| What happens when something breaks (support, recourse) | **No**, but it affects how long a problem lasts. |
| **How reliably the server stays on** | **YES.** A server that is off does not refresh the price grid, and stale grid data is the largest measured cause of pricing error in this system. |

So the decision is **not** "cheap vs accurate". It is **"how much do we want to
pay to reduce the risk of the machine being unavailable?"**

**Recommendation: Contabo Cloud VDS S, in Mumbai, on the 1-month term first.**
It is about **half** of Azure's cost, has dedicated CPU cores (so no speed
penalty), and sits in the same city as the client. The one real concession is a
weaker uptime guarantee. Take it monthly at $72.55 for the parallel run — the
saving from committing to 24 months is only about $9/month, which is not worth
locking in before the machine has proven itself.

### What to order

| Setting | Choose |
|---|---|
| Plan | **Cloud VDS S** — 6 dedicated cores, 24 GB RAM, 180 GB NVMe |
| Term | **1 month** ($47.00) — switch to 12/24 months after the trial |
| Region | **Asia (India)** (+$25.55) |
| Storage | **180 GB NVMe (included)** — do NOT add the 1 TB option, we use ~10 GB |
| Image | **Ubuntu** (included) — must be **24.04 LTS** |
| Windows | **No** (+$61/month, and all our scripts are Linux) |
| cPanel / Plesk | **No** — we do not use them |

---

## 1. The three realistic options

All prices are for a **Linux** server. Confirm them before ordering — they move,
and Contabo's headline rates assume a long commitment.

| | **Azure `D4s_v5`** | **Contabo Cloud VPS 8** | **Contabo Cloud VDS S** |
|---|---|---|---|
| CPU | 4 vCPU, **dedicated** | 8 vCPU, **shared** | 6 cores, **dedicated** (AMD EPYC) |
| RAM | 16 GB | 24 GB | 24 GB |
| Disk | 256 GB SSD | 300 GB SSD | 180 GB **NVMe** |
| Location | Central India (Pune) | Navi Mumbai | Navi Mumbai |
| **Cost / month** | **~₹12,400 (~$145)** | ~$19 + India fee | **$63–73** (see below) |
| Uptime guarantee | **99.9%** (with credits) | **95%** | **95%** |
| Term for that price | Pay-as-you-go | 24-month commitment | any term, no setup fee |

### Corrected pricing — from Contabo's actual order page

An earlier draft of this document estimated the India surcharge at about €3/month.
**That was wrong.** The location fee scales with the plan tier: €2.40 is the
entry-level VPS figure, and for Cloud VDS S it is **$25.55/month**. The real
totals are:

| Term | Server | India location fee | **Total / month** |
|---|---|---|---|
| **1 month** | $47.00 | $25.55 | **$72.55** |
| 12 months | $39.95 | $25.55 | **$65.50** |
| 24 months | $37.60 | $25.55 | **$63.15** |

No setup fee on any term. So Contabo VDS S is **about half of Azure's ~$145**,
not one third. The saving is roughly **$70–80/month (~$900/year)** — still
material, but smaller than first stated.

**Exact hardware on offer:** 6 dedicated cores of **AMD EPYC 7282 @ 2.8 GHz**,
24 GB RAM, 180 GB NVMe, **250 Mbit/s** port, Asia (India) region at 21 ms
latency, Ubuntu included at no cost.

**Windows Server on Contabo costs +$61/month**, which would take the total to
~$133 — essentially Azure's price for weaker guarantees. On Contabo, Windows
throws the saving away.

**Disk size note:** we currently use 1.1 GB, growing ~1.6 GB/year. Even the
smallest option (180 GB) is many years of headroom. Disk size is not a
differentiator here.

**RAM note:** we specified 16 GB because the nightly rebuild peaks at 1–2 GB on
top of the model and database. Both Contabo options give **24 GB**, which is more
headroom than the Azure box we specced.

---

## 2. Will performance actually be the same?

This is the part where it matters *which* Contabo product is chosen — the two are
very different machines.

**Contabo Cloud VPS (the €14 one) uses shared CPU.** Independent testing scores
Contabo's shared-CPU instances around **482 single-core points, noticeably behind
higher-tier providers**, with weaker memory and network throughput. Against our
two workloads:

| Our workload | Depends on | On shared-CPU VPS |
|---|---|---|
| **Nightly rebuild** (5–9 min today) | Multiple cores, runs at 02:30 | **Fine.** Even 2–3× slower is irrelevant at 2am. |
| **One live price** (1–3 sec today) | **Single-core speed** | **Slower — likely 2–4 sec.** The desk waits for this, so they would notice. |
| Disk | Barely | Fine. Our files total ~180 MB and the database writes a few thousand rows a day. |
| Network to their data | Latency | **Fine.** Navi Mumbai to Azure Pune/Mumbai is a few milliseconds. |

**Contabo Cloud VDS (the €39 one) uses dedicated cores.** This removes the single
issue above — the reason the cheap VPS feels slow is that its cores are shared
with other customers, and VDS does not share them. This is why the recommendation
is VDS and not the cheapest VPS.

> **Being honest about what we know:** we have **not** benchmarked our engine on
> Contabo hardware. The figures above are from published third-party testing of
> Contabo generally, not from running our model. Section 7 describes how to find
> out for real, cheaply, before committing.

---

## 3. The reliability question — the one that touches accuracy

This is the crux, and it deserves plain treatment rather than a sales pitch in
either direction.

**The contractual guarantees are genuinely far apart:**

| | Guarantee | Downtime that allows |
|---|---|---|
| Azure (single VM, premium SSD) | **99.9%** | ~8.8 hours/year, with service credits |
| Contabo (both VPS and VDS) | **95%** | **~18 days/year**, with little practical recourse |

**But the observed behaviour is much better than the guarantee.** Contabo is
widely reported as reliable in day-to-day use — one published review logged
**100% uptime over 50 days of monitoring**. The 95% figure is a contractual
floor, not an expectation. It is what you can *hold them to*, not what you should
*expect*.

**Why this matters more for us than for a typical website:** our own measurements
show pricing error grows with grid staleness — from about **2 points** on
fresh data to about **3 points** at 8–14 days old. The entire reason for buying a
server was that the Windows laptop missed 43% of its nightly runs.

**What softens it considerably:** the nightly timer is configured with
`Persistent=true`, meaning **if the machine was off at 02:30, the job runs as soon
as it comes back**. So a few hours of downtime costs nothing — the job simply
catches up. The genuine risk is a **multi-day** outage, because the client's grid
API only serves a short recent window, and a day that passes unrecorded is lost
permanently.

**Net assessment:** an occasional short Contabo outage is a non-event for us. A
multi-day outage would cost grid history we cannot get back. That risk is real
but low, and it is the thing the Azure premium is actually buying.

---

## 3b. Backups — server-to-server, and the one trap in it

**The push-out model is correct.** Our server sends the files OUT to storage on a
schedule. Nothing connects inward to us, no firewall port is opened, and there is
no inbound attack surface. `backup.sh` already works this way.

**The trap: do not put the backup at the same provider as the server.** Contabo
Object Storage is the cheapest and easiest option, but a backup that shares a
provider with the thing it is protecting fails together with it. A Contabo
account problem, a billing lapse or a regional incident takes both.

**Recommended destinations, in order:**

| Destination | Cost for our volume | Independent of the server? |
|---|---|---|
| **Backblaze B2** | ~$1–2/month | **Yes** — different company entirely |
| **Azure Blob Storage** | ~$1–5/month | **Yes** — and they already have Azure |
| Contabo Object Storage | ~$3/month | **No** — same provider. Convenient, weaker. |
| Their own in-house server | — | Yes, but see below |

**On copying into their building:** do not have our server push into their
network — that needs an inbound firewall opening on their side, which is exactly
what they wanted to avoid. Have **their** machine pull from the object storage
instead. Outbound-only at both ends.

**What changes in the code:** nothing structural. `backup.sh` now supports any
S3-compatible or cloud destination through `rclone`, so Contabo Object Storage,
Backblaze B2, AWS S3 and Azure Blob are all one environment variable:

```
GS_BACKUP_RCLONE_REMOTE=b2:glowstar-backups
```

Two improvements were made while adding this, both tested:

- **Only the current night's files are sent.** The previous version re-uploaded
  the whole 30-day directory every night — about 600 MB of transfer for 20 MB of
  new data.
- **A failed off-site copy now fails loudly** and returns a non-zero exit, so
  systemd records it. It previously could not silently "succeed" with no
  destination configured either — that case now prints a clear warning. A backup
  everyone believes in but which is not actually leaving the machine is the
  failure people discover on the day they need it.

**Volume to expect:** roughly 20–25 MB per night (the gzipped grid history plus a
small database dump), so a few GB a year. Any of the options above costs a
handful of dollars a year at that size.

---

## 4. Support — the factor that is easy to underrate


| | Azure | Contabo |
|---|---|---|
| Support model | Paid tiers with response-time commitments | Ticket-based, slower |
| Who fixes a 9am problem | Microsoft, then us | **Us**, mostly |
| Escalation path | Formal | Limited |

---

## 5. Linux vs Windows — a separate decision

This is independent of the provider. Both offer both.

### Linux (recommended, and already agreed)

**Pros**
- Everything is **already written and tested** — the installer, the three service
  definitions and the backup job have been run end-to-end on a clean Ubuntu
  machine including a full backup and restore.
- **`Persistent=true` scheduling** — if the machine was off at the scheduled time,
  the job runs on the next boot. This directly addresses the failure that has
  already cost us accuracy.
- **Cheaper on both providers** — Windows adds ~40% on Azure; on Contabo it is a
  paid licence add-on that erodes much of the saving.
- Uses ~1.5–2 GB less RAM at idle.
- Standard for Python/ML services; libraries are tested there first.

**Cons**
- The client's team has no Linux experience. Mitigated by `OPERATIONS.md`, but a
  real onboarding cost.
- Fewer people in their building can help in an emergency.

### Windows

**Pros**
- Their team can navigate it today.
- Familiar tooling and remote desktop.

**Cons**
- **All deployment scripts must be rewritten** — 2–4 days, and the new versions
  would be unproven where the current ones are tested.
- Adds cost on both providers.
- **Task Scheduler is where this project already got burned** — it silently killed
  the nightly job for 10 consecutive nights (error `0xC000013A`). A Windows
  *Server* would not sleep like that laptop did, but the failure was silent, and
  silence is the dangerous part.

**Verdict:** Linux, on either provider. The scripts exist and are tested; Windows
means paying more for less-proven automation.

---

## 6. Pros and cons, side by side

### Azure

**Pros**
- 99.9% uptime guarantee with service credits.
- Enterprise vendor: clean GST invoicing in India, formal contracts, procurement
  teams are comfortable with it.
- Paid support tiers with response commitments.
- Same cloud as the client's inventory API — trivially low latency, no
  cross-provider egress charges.
- Resizing and snapshots are instant and self-service.
- Managed PostgreSQL and Blob Storage available as drop-in upgrades later.

**Cons**
- **~$145/month — roughly 3× the VDS option and 7× the VPS one.**
- 4 vCPU / 16 GB is less raw resource than either Contabo option.
- Egress is chargeable (small for us, but non-zero).

### Contabo Cloud VDS (recommended)

**Pros**
- **~$45/month — about one third of Azure**, saving roughly **$1,200/year**.
- **Dedicated AMD EPYC cores** — no shared-CPU slowdown, so live pricing stays
  responsive.
- **NVMe storage** (faster than the Azure SSD we specced).
- 24 GB RAM — more headroom than we asked for.
- **Navi Mumbai** — same city as the client, low latency to their systems.
- Generous included bandwidth; no per-GB egress anxiety.

**Cons**
- **95% contractual uptime** — weak recourse if they have a bad month.
- Slower, ticket-based support with no response guarantee.
- Best price needs a **24-month commitment**.
- Budget provider — worth confirming their finance team accepts the vendor and
  the invoicing.
- Fewer managed services to grow into later (no managed PostgreSQL equivalent).

### Contabo Cloud VPS (the cheapest option)

**Pros**
- **~$19/month.** By far the cheapest.
- 24 GB RAM, 300 GB SSD.

**Cons**
- **Shared CPU** — the live price request is the one thing that would visibly
  slow down, and it is the thing the desk actually waits for.
- Same 95% guarantee and support limitations as VDS.
- **We would not recommend this as the permanent home**, only as a trial
  (section 7).

---

## 7. How to decide this properly, for about $19

We do not have to guess at the performance question. The parallel-run period was
already part of the plan — the desk uses the engine alongside Excel for about two
weeks before switching.

**Proposed approach:**

1. **Take a Contabo VDS (or even the cheap VPS) for one month, no long commitment.**
2. Deploy with the existing scripts. **Nothing in `deploy/` is Azure-specific** —
   it is a plain Ubuntu server in every case.
3. During the parallel run, measure two things that actually matter:
   - **Live price response time** — is it 1–3 seconds or 4+?
   - **Did the nightly job run every night?** (`systemctl list-timers`, which is
     the weekly check in `OPERATIONS.md` anyway.)
4. **Then decide with evidence instead of opinion.** If it holds up, commit to the
   longer term and keep the saving. If it does not, move to Azure — that is a
   redeploy of the same scripts, roughly half a day, not a rewrite.

This costs one month's rental to remove all the guesswork, and it fits inside a
period we had already planned to run cautiously.

**One thing that must be set up either way:** off-server backups. On Contabo we
would use their Object Storage rather than Azure Blob — the same backup script,
a different destination, one line in `.env`.

---

## 8. What we know, what we researched, and what we do not know

Stated plainly, so nothing here is mistaken for more certainty than it has.

| Claim | Basis |
|---|---|
| Same code produces the same prices on any of these servers | **Certain** — it is the same software and the same data |
| Grid staleness costs ~1 point of accuracy per week | **Measured** on our own data |
| Deployment scripts work on clean Ubuntu, including backup and restore | **Measured** — run end-to-end |
| A missed night is recovered automatically on next boot | **Measured** — `Persistent=true`, verified with systemd |
| Contabo pricing, locations, dedicated-core VDS | **Researched** from Contabo's own site, August 2026 |
| Contabo shared-CPU is slower than premium providers | **Researched** — third-party benchmarks, not our workload |
| Contabo's contractual uptime guarantee is 95% | **Researched** — their published terms |
| Contabo is reliable in practice despite that figure | **Researched** — third-party monitoring, not our experience |
| **How fast OUR engine runs on Contabo** | **Not known.** Only a trial answers this. |
| **Contabo's real-world support responsiveness for us** | **Not known.** |

---

## 9. Bottom line

- **The prices we deliver will be identical** on any of these machines. This is not
  a trade-off between cost and accuracy.
- **The real trade-off is availability insurance**, and the nightly job's
  catch-up behaviour already absorbs the common failure.
- **Recommendation: Contabo Cloud VDS S in Mumbai, on Linux**, proven during the
  parallel run before committing to a long term.
- **Choose Azure instead if** the client's policy requires a formal SLA, their
  finance team prefers an enterprise vendor, or they want somebody other than us
  to be accountable when the machine itself fails. Those are all legitimate
  reasons, and $145/month is a fair price for them.
- **Do not choose the cheapest shared-CPU VPS as the permanent home.** The saving
  over VDS is about $25/month, and it is paid for with the response time the desk
  experiences on every single stone.

---

### Sources

- [Contabo Cloud VPS plans and pricing](https://contabo.com/en/vps/)
- [Contabo Cloud VDS — dedicated cores](https://contabo.com/en/vds/)
- [Contabo Mumbai data centre](https://contabo.com/blog/contabo-will-launch-a-new-data-center-in-india-in-2024/)
- [Contabo performance review — Cybernews](https://cybernews.com/best-web-hosting/contabo-review/)
- [Contabo review — EXPERTE.com](https://www.experte.com/server/contabo)
- [Contabo pricing guide — Cybernews](https://cybernews.com/best-web-hosting/contabo-review/pricing/)
- [Azure Linux Virtual Machines pricing](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/)
- [Azure pricing calculator](https://azure.microsoft.com/en-us/pricing/calculator/)
- [Azure SLA documentation](https://learn.microsoft.com/en-us/azure/reliability/concept-service-level-agreements)
