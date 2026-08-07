# Server Requirements — Full Breakdown

**For:** Glow Star (Milan Bhai, Jay Bhai)
**From:** Lets Automation
**Purpose:** exactly what hardware is needed, for which parts of the programme,
what it costs, and why each choice was made.

---

## 1. Short version

**One server is needed — and it is needed now.**

| | |
|---|---|
| **Runs** | Pricing Engine, Inventory Intelligence, Gap Engine |
| **MOU workstream** | A, B, C |
| **Needed** | **Now** — this is the current blocker |
| **Where** | Azure (public cloud) |
| **OS** | Ubuntu 24.04 LTS (recommended) |
| **Size** | 4 vCPU / 16 GB RAM / 256 GB |
| **Cost** | ~$140–170 / month |

**Order it now.** Everything else in the pricing programme is finished and
tested; the server is the only thing standing between the current state and live
operation.

---

## 2. What the server connects to

The server (Workstreams A, B, C) talks only to the public internet:

| Data source | Address |
|---|---|
| Your inventory + sales | `channelpartnerapi.azurewebsites.net` |
| Your price grid | `pricingapi.diamanto.co` |
| Market comparables | `app.uni.diamonds` |

All three are public HTTPS endpoints. This server never needs to reach anything
inside your building.

---

## 3. The server to order now

### 3.1 Specification

| Item | Requirement | Why this number |
|---|---|---|
| **Provider** | Microsoft Azure | Your inventory API is already on Azure. Same network = fastest, and no cross-cloud data charges. |
| **Region** | Same as your Channel Partner API | Lowest latency on the data we pull every night. |
| **Size** | 4 vCPU, 16 GB RAM (Azure `D4s_v5` or equivalent) | Measured, see 3.2. |
| **Disk** | 256 GB SSD | Currently 1.1 GB used, growing ~1.6 GB/year. 3+ years of headroom. |
| **OS** | **Ubuntu 24.04 LTS** (recommended — see section 4) | |
| **Uptime** | **Must never sleep or shut down** | This is the single biggest accuracy factor. See 3.3. |
| **Network** | Port 443 inbound (your CRM → us). Outbound HTTPS. | |
| **Access** | SSH for our team | For deployment, monitoring and fixes. |

### 3.2 Why 16 GB and not 8 GB

Measured on the real system:

| Task | Memory |
|---|---|
| Price grid held in memory | ~600 MB |
| Inventory + sales data | ~50 MB |
| Nightly model rebuild (peak, transient) | 1–2 GB |
| API service (2 workers) | ~2 GB |
| PostgreSQL | ~2 GB |
| Operating system | ~1 GB |

**8 GB would work most of the time and fail during the nightly rebuild** — which
is exactly when nobody is watching. 16 GB removes that risk for a small cost
difference.

Timings, measured: nightly rebuild 5–9 minutes; a price request 1–3 seconds; a
5,000-stone batch a few minutes.

### 3.3 Why "always on" is not a nicety

**This is the most important line in this document.**

The engine refreshes your price grid every day. We measured what happens when
that refresh is late:

| How current the grid data is | Our pricing error |
|---|---|
| **0–3 days old** | **1.5 points** |
| 4–7 days | 2.0 points |
| 8–14 days | 3.0 points |
| **15–30 days old** | **4.3 points** |

**Stale data costs nearly 3 points of accuracy** — more than any modelling change
we have made in this entire project.

**And this is not hypothetical. It already happened.** The jobs currently run on a
Windows laptop. Over a recent 23-day window they ran on only **13 days — 10 nights
missed, 43% of the window.** The laptop went on battery or to sleep, and Windows
Task Scheduler silently killed the job (error `0xC000013A`). Nothing alerted
anyone; the nightly retrain log contains a handful of entries where it should have
hundreds.

A proper always-on server eliminates this entire class of problem.

### 3.4 What runs on the server

One machine, everything:

1. **The pricing API** — answers your CRM (`/frontoffice/price`, `/decision`, etc.)
2. **The nightly job** — 02:30: pull fresh data → rebuild the model → test it →
   promote only if it scores better → back up
3. **PostgreSQL** — every price we quote, every decision your desk makes, every
   AI score (the audit trail)
4. **Later: Inventory Intelligence and the Gap Engine** — same data, same
   pipeline, **no upgrade needed**

There is no separate "AI server" or "model server". The model is a 9 MB file this
machine rebuilds each night.

---

## 4. Can the server be Windows instead of Linux?

**Yes. It genuinely works.** The engine runs on Windows today — that is exactly
how it is running now. Nothing about the software requires Linux. So this is a
cost-and-operations decision, not a technical impossibility.

Here is the honest comparison.

### Reasons to choose Linux (our recommendation)

| | Detail |
|---|---|
| **Cost** | Windows Server licensing adds roughly **40% to the monthly bill** — about **$60–80/month extra**, ~$800/year, for the same computing power. |
| **Scheduling reliability** | Linux's scheduler (`systemd`) has a built-in "if the machine was off at 02:30, run as soon as it starts" behaviour, and logs every run centrally. Windows Task Scheduler has an equivalent setting but it is easier to get wrong — and on this project it **already failed silently for 10 nights.** (In fairness: that was a *laptop* with battery and sleep settings. A Windows *Server* in Azure would not sleep. But the failure was silent, and that is the part that worries us.) |
| **Ready to deploy today** | Our installation script, service definitions and backup job are written for Linux and have been **run end-to-end on a clean Ubuntu 24.04 machine**, including a full database backup and restore. Choosing Windows means rewriting all of them — new, unproven scripts and a slower first deployment. |
| **Memory efficiency** | Windows Server uses roughly 1.5–2 GB more RAM at idle, which matters during the nightly rebuild. |
| **Standard for this kind of work** | Python data/ML services are overwhelmingly deployed on Linux; libraries are tested there first. Fewer surprises. |

### Legitimate reasons to choose Windows anyway

- **Your IT team only supports Windows.** A server your team cannot confidently
  manage is worse than a slightly more expensive one they can. This is a good
  reason and we would support it.
- **Company policy** requires Windows.

### What changes if you pick Windows

- Add roughly **$60–80/month**.
- Add **2–4 days** to first deployment while we rewrite the service and scheduling
  scripts for Windows.
- Everything else is identical — same engine, same API, same accuracy.
- We would then insist on **explicit monitoring of the nightly job**, because that
  is the specific thing that failed before.

**Our recommendation: Ubuntu Linux.** But tell us if Windows suits
your team better — it is a supportable choice and we will make it work properly.

---

## 5. Database

**PostgreSQL, installed on the same server** (no separate database server needed).

It stores three things:

| Table | Why it exists |
|---|---|
| **Quotes** | Every price we have ever published — which stone, what we said, which model version said it. This is what answers "what did you quote in March, and why?" |
| **Decisions** | Every accept / reject / override from your desk, with the reason where one is required |
| **Scores** | The AI score components, so the scoring can be improved later against what stones actually did |

**Alternative:** Azure's managed PostgreSQL adds ~$50–80/month and takes backups
and patching off our hands. Worth it if you would rather not rely on us for that.
Our recommendation is to start with it on the same server and move later if you
prefer.

---

## 6. Backups — one thing that cannot be recovered

Most of our data can be rebuilt by re-downloading it. **One file cannot.**

`history.json` records what every cell of your price grid read **on every past
day**. Your grid API only serves a short recent window, so once a day passes
unrecorded, that day is gone permanently. Without this history the pricing model
cannot be honestly tested — and it is 106 MB today and growing.

The nightly job backs it up automatically, along with the database. **We need one
thing from you: somewhere off the server to copy it to** — an Azure storage
account is simplest. A backup on the same machine is not a backup.

---

## 7. Cost summary

| Item | Monthly |
|---|---|
| Server (Linux, 4 vCPU / 16 GB / 256 GB) | $140–170 |
| Off-server backup storage | $5–15 |
| **Total to go live now** | **~$150–185** |
| *Windows instead of Linux* | *+$60–80* |
| *Managed PostgreSQL instead of self-hosted* | *+$50–80* |

Please confirm current pricing with Azure — these are close estimates, not quotes:

- **Pricing calculator:** https://azure.microsoft.com/en-us/pricing/calculator/
- **Linux VM prices:** https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/
- **Managed disks (the 256 GB SSD):** https://azure.microsoft.com/en-us/pricing/details/managed-disks/
- **Blob storage (backups):** https://azure.microsoft.com/en-us/pricing/details/storage/blobs/

In the calculator: *Virtual Machines* → your region → **Linux** → size **D4s_v5**
→ **256 GB Premium SSD**, then *Storage Accounts* → **Blob** → ~50 GB.
A **1-year reserved instance is roughly 30–40% cheaper** than the pay-as-you-go
rate shown, which is worth considering for a machine that runs permanently.

---

## 8. What we need from you

**To go live with pricing:**

1. **The server**, specified in section 3, with SSH access for our team.
2. **Confirm the OS choice** — Linux (recommended) or Windows.
3. **A storage location for backups** (section 6).
4. **Confirmation that your CRM can call an external HTTPS endpoint** — any
   firewall rule needed on your side?
5. **Who should be alerted** if the system reports a problem.

---

## 9. What happens after you provide the server

| Day | What we do |
|---|---|
| 1 | Install and configure everything; hand you the API key |
| 2 | Load historical data; verify prices against stones you already know |
| 3–4 | You connect a test screen; we check together |
| Then | Your desk uses it live alongside the current process for ~2 weeks |
| Then | Excel retires |

Days 1–2 are largely automated — the installation is a single scripted command,
already written and tested.
