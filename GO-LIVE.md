# Going Live — What We Need From You

**For:** Glow Star IT
**From:** Lets Automation
**Date:** 2026-08-10

---

## Where we are

**The pricing engine is installed and running on the new server.** It is pulling
your live inventory, refreshing your price grid nightly, rebuilding and testing
the model every night at 02:30 India time, and recording every price it produces
for audit.

Everything below is about **connecting it to your CRM**. The system itself is
finished and working.

| | Status |
|---|---|
| Server built, secured, firewalled | Done |
| Pricing engine installed and running | Done |
| Live inventory feed connected (27,606 records) | Done |
| Price grid refreshing nightly | Done |
| Model rebuilt, tested and promoted automatically | Done |
| Every quote recorded to the database for audit | Done |
| Nightly backups | Done |
| **Reachable by your CRM** | **Needs item 1 below** |

---

## 1. A web address for the service — the one thing blocking go-live

Your CRM needs a proper web address to call. Today the service can only be
reached from the server itself, which is deliberate: without a web address we
cannot install an HTTPS certificate, and without HTTPS your API key would travel
across the internet unencrypted.

**What we need you to create: one DNS "A record".**

| Field | Value |
|---|---|
| **Type** | `A` |
| **Name / Host** | `pricing` (giving `pricing.yourdomain.com`) |
| **Value / Points to** | `217.217.248.111` |
| **TTL** | Default (or 300) |
| **Proxy / CDN** | **OFF** — see the note below |

**Where to do it:** wherever your domain's DNS is managed — usually your domain
registrar (GoDaddy, BigRock, Namecheap) or your website host. Whoever looks after
your company website will know. It takes about five minutes.

**Please tell us the exact address you have created**, e.g.
`pricing.glowstar.com`.

> **Two technical notes for whoever makes the change:**
>
> 1. **If you use Cloudflare, please leave the record "DNS only" (grey cloud),
>    not proxied (orange cloud).** A proxied record hides the server's real
>    address and prevents the certificate from being issued.
> 2. **Port 80 must stay open to the internet.** It only redirects visitors to
>    the secure HTTPS address and serves no data — but the free certificate
>    renews itself through an automated check on port 80. If that port is closed,
>    everything works for about 60 days and then stops. If your security policy
>    cannot allow this, tell us and we will use a different verification method
>    instead.

### If you would rather not use your own domain

We can host the address on our domain instead (for example
`glowstar.letsautomation.in`) and have it live **today**, with no action needed
from you. Your own domain looks better long term and is not tied to us — but if
arranging the DNS change will take time, say the word and we will start on ours
and move it later. Moving it afterwards is a small change.

---

## 2. Confirm your CRM can call an external address

Your CRM will make a normal outgoing HTTPS request to the address above, exactly
like any other external service it already uses.

**Please confirm:** does anything on your network need to allow this? Some
companies run an outbound proxy with an approved-sites list. If yours does, add
the address from item 1.

Nothing needs to be opened *inward* to your network. We never connect to you.

---

## 3. Where should backups be stored?

The system takes a backup every night. One of those files **cannot be recreated
if it is lost** — it is the day-by-day record of what every cell of your price
grid read on every past date. Your grid system only serves recent dates, so once
a day passes unrecorded it is gone permanently, and the pricing model depends on
that history.

Right now those backups sit on the same server that produces them. That is not a
real backup: if the server fails, both copies are lost together.

**We need somewhere off the server to copy them to.** Any of these works, and it
is a two-minute change at our end:

| Option | Approx. cost | Notes |
|---|---|---|
| **Azure Storage account** | ~$1–5/month | Simplest if you already use Azure |
| Backblaze B2 | ~$1–2/month | Cheapest |
| A folder on a server you already run | — | Fine, if it is a different machine |

Please either create one and send us the access key, or tell us to set one up and
bill it through.

*(As a temporary measure we are keeping a manual copy off the server, but this
should be automated soon.)*

---

## 4. Who should we contact if something goes wrong?

Please give us **one name and one email address** for alerts — for example if the
nightly job fails or the server has a problem. It will not be noisy; this is for
genuine faults only.

---

## What happens after you send us item 1

| Step | Who | How long |
|---|---|---|
| Point the service at your address, install the HTTPS certificate | Us | ~30 minutes |
| Test a real price over the internet | Us | 10 minutes |
| Send you the address and your API key, securely | Us | Same day |
| Your team connects the CRM and tries a few stones | You | Your pace |
| Run alongside your current Excel process | Both | ~2 weeks |

**Your CRM will need two things from us, which we will send once the address is
live:** the web address, and a secret API key that must be included with every
request. We will send the key by a secure method — **not** over email or
WhatsApp.

---

## Summary — what we need

1. **A DNS A record** pointing a name like `pricing.yourdomain.com` at
   `217.217.248.111` — *this is the only thing blocking go-live*
2. **Confirmation** your CRM can call an external HTTPS address
3. **A storage location** for off-server backups
4. **A contact** for fault alerts

Item 1 is the blocker. Items 2–4 can follow.
