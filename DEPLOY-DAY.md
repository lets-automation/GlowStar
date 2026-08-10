# Deployment Day — Step by Step

**Server:** Contabo Cloud VDS S, Ubuntu 24.04, Asia (India)
**Run these in order.** Each step says what "good" looks like, so you know
whether to continue or stop.

> **Before starting, have these ready:**
> - Server IP address and the root password from Contabo
> - The `.env` credentials (Channel Partner, Diamanto, UNI)
> - A Backblaze B2 (or Azure Storage) account for off-site backups
> - The domain name, if you are doing TLS today
>
> **Time:** about 45 minutes, most of it waiting for `pip install`.

---

## Step 0 — Confirm the server is what we ordered

From your Windows machine (Windows Terminal or PowerShell):

```
ssh root@<SERVER_IP>
```

Type `yes` at the fingerprint prompt, then the password. Once connected:

```bash
lsb_release -a && nproc && free -g && df -h / && uname -m
```

✅ **Good:** `Ubuntu 24.04`, `6` CPUs, ~24 GB RAM, ~180 GB disk, `x86_64`.

🛑 **Stop if** it says Ubuntu 22.04 — rebuild the server with 24.04 from the
Contabo panel first. 22.04's Python is a release candidate.

---

## Step 1 — Basic hardening (10 minutes, do not skip)

This server has a public IP and the client declined an IP allowlist, so it is
reachable from anywhere.

```bash
# 1a. Change the root password Contabo emailed you
passwd

# 1b. System updates
apt-get update && apt-get upgrade -y

# 1c. Firewall — only SSH and web
apt-get install -y ufw
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
ufw status
```

✅ **Good:** `Status: active` with 22, 80 and 443 allowed.

**Note on port 80:** leave it open. The TLS certificate renews via an inbound
check on port 80, and closing it makes the certificate silently expire about 60
days later. Port 80 only ever redirects to HTTPS; it serves no data.

**Recommended — SSH keys instead of passwords.** From *your Windows machine*, in
a **new** terminal (not the SSH session):

```
ssh-keygen -t ed25519
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh root@<SERVER_IP> "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

Test that `ssh root@<SERVER_IP>` now connects without a password **before**
disabling password login.

---

## Step 2 — Send the files up

`deploy-bundle.tar.gz` (20 MB) is in the project folder. It contains the code,
the Rapaport sheets, the current model, and — most importantly — **the grid
history, which cannot be rebuilt if lost.**

From **your Windows machine**, in a terminal in the project folder:

```
scp deploy-bundle.tar.gz root@<SERVER_IP>:/root/
```

Then back in the SSH session on the server:

```bash
mkdir -p /opt/glowstar
tar xzf /root/deploy-bundle.tar.gz -C /opt/glowstar
ls /opt/glowstar
du -sh /opt/glowstar/data/master_grid/history.json
```

✅ **Good:** you see `glowstar/`, `deploy/`, `CSV2_ROUND_8_4.csv`,
`records.json`, and `history.json` is about **102 MB**.

🛑 **Stop if** `history.json` is missing or tiny. Do not continue — that file is
irreplaceable and the grid model depends on it.

---

## Step 3 — Create `.env` (the credentials)

**This must exist before the installer runs.** The API service will not start
without it.

```bash
cd /opt/glowstar
cp .env.example .env
nano .env
```

In `nano`: edit, then **Ctrl+O**, **Enter** to save, **Ctrl+X** to exit.

Fill in:

| Setting | Value |
|---|---|
| `GS_ENV` | `production` |
| `GS_API_KEY` | Generate one — see below. This is what the CRM will send. |
| `GS_DB_PASSWORD` | Invent a strong password |
| `GS_DATABASE_URL` | `postgresql+psycopg://glowstar:<that same password>@localhost/glowstar` |
| `CHANNEL_PARTNER_*`, `DIAMANTO_*`, `UNI_*` | The real credentials |
| `GS_RATE_LIMIT` | `120` (leave as is) |
| `GS_BACKUP_RCLONE_REMOTE` | Leave blank for now — Step 7 |

Generate the API key:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Then lock the file down:

```bash
chmod 600 /opt/glowstar/.env
```

⚠️ **The password inside `GS_DATABASE_URL` must exactly match `GS_DB_PASSWORD`.**
A mismatch here is the most common first-deployment failure.

---

## Step 4 — Run the installer

```bash
cd /opt/glowstar
export GS_DB_PASSWORD='<the same password you put in .env>'
bash deploy/install.sh
```

This takes 5–10 minutes (mostly `pip install`). Watch for these lines:

```
==> timezone
  timezone: IST +0530  (Asia/Kolkata)
==> preflight: the Rapaport list must be on disk BEFORE the API starts
  ok: Rap list and .env present
```

✅ **Good:** it finishes printing "Remaining, by hand" with four items.

🛑 **If it stops at the timezone check**, the server did not switch to India
time. Do not continue — the nightly job would run at 08:00 instead of 02:30.

🛑 **If it stops at the preflight**, a file is missing. It tells you which.

---

## Step 5 — Verify it actually works

Two checks, and **the second one is the one that matters**.

```bash
# 5a. Is the service running?
systemctl status glowstar-api --no-pager
curl localhost:8000/health
```

✅ **Good:** `active (running)`, and health returns a model version.

```bash
# 5b. Does it PRICE? (health returns 200 even when pricing is broken)
curl -s -X POST localhost:8000/price \
  -H "X-API-Key: <your GS_API_KEY>" \
  -H 'Content-Type: application/json' \
  -d '{"Shape_full":"ROUND","Weight":1.01,"Color":"G","Clarity":"VS1"}'
```

✅ **Good:** JSON containing `suggested_discount` — a negative number around
-45 to -55.

🛑 **If 5a passes but 5b fails, the deployment is not done.** That is exactly the
failure mode we designed the preflight around. Send us the output.

```bash
# 5c. Confirm the nightly job is scheduled
systemctl list-timers glowstar-nightly --no-pager
```

✅ **Good:** `NEXT` shows tomorrow **02:30 IST**. If it shows 02:30 UTC or 08:00,
stop — the timezone did not apply.

---

## Step 6 — Domain and HTTPS

Only after the DNS **A record** for your domain points at this server's IP.
Check it has propagated first:

```bash
dig +short pricing.<yourdomain>.com
```

✅ It should print this server's IP. If not, wait — DNS can take up to an hour.

```bash
# Put the real domain into the nginx config
sed -i 's/YOUR_DOMAIN/pricing.yourdomain.com/g' /opt/glowstar/deploy/nginx.conf
cp /opt/glowstar/deploy/nginx.conf /etc/nginx/sites-available/glowstar
ln -sf /etc/nginx/sites-available/glowstar /etc/nginx/sites-enabled/glowstar
rm -f /etc/nginx/sites-enabled/default
nginx -t
```

✅ **Good:** `syntax is ok` and `test is successful`.

```bash
certbot --nginx -d pricing.yourdomain.com
systemctl reload nginx
```

Then test from **outside** the server (your own machine):

```
curl -s -X POST https://pricing.yourdomain.com/price -H "X-API-Key: <key>" -H "Content-Type: application/json" -d "{\"Shape_full\":\"ROUND\",\"Weight\":1.01,\"Color\":\"G\",\"Clarity\":\"VS1\"}"
```

✅ **Good:** a price comes back over HTTPS. **This is the URL the CRM will use.**

---

## Step 7 — Off-site backups

A backup on the same disk is not a backup. Use a **different provider from
Contabo** — if the Contabo account has a problem, both would be lost.

```bash
rclone config
```

Follow the prompts: `n` for new remote, name it `b2`, choose Backblaze B2 (or
`s3` / Azure Blob), paste the account keys. Then:

```bash
nano /opt/glowstar/.env
# set:  GS_BACKUP_RCLONE_REMOTE=b2:glowstar-backups
```

Test it immediately — do not wait for tonight:

```bash
sudo -u glowstar bash -c 'set -a; . /opt/glowstar/.env; set +a; /opt/glowstar/deploy/backup.sh'
```

✅ **Good:** `off-site copy sent to b2:glowstar-backups`.

🛑 **If you see** `WARNING: no off-site destination configured`, the variable did
not load. Fix it now — this is the one failure people discover on the day they
need the backup.

---

## Step 8 — First full run and handover

```bash
# Run the whole nightly sequence once, by hand, while you are watching
sudo systemctl start glowstar-nightly
journalctl -u glowstar-nightly -f
```

Press **q** to exit the log view. It takes 5–15 minutes.

✅ **Good:** the snapshot, the retrain and the backup all complete. The retrain
may say the candidate was **not promoted** — that is the promotion gate working
correctly, not an error.

**Final checklist before telling the client it is live:**

- [ ] `curl` over HTTPS from outside returns a real price
- [ ] `systemctl list-timers` shows tomorrow 02:30 **IST**
- [ ] `ls /var/backups/glowstar` has today's files
- [ ] The off-site copy landed (check the B2/Azure console)
- [ ] `.env` is `chmod 600`
- [ ] API key sent to the client **securely** — not over email or WhatsApp
- [ ] `ufw status` is active
- [ ] They have `OPERATIONS.md`

---

## If something goes wrong

Collect this and send it to us — it is usually enough to diagnose without
logging in:

```bash
systemctl status glowstar-api --no-pager
systemctl list-timers glowstar-nightly --no-pager
journalctl -u glowstar-api -n 50 --no-pager
journalctl -u glowstar-nightly -n 50 --no-pager
df -h /
free -g
date
```

**Do not restart the service before collecting this.** Restarting clears the
evidence.

---

## Day 2 onwards

Hand the client `OPERATIONS.md`. The only recurring task is the **weekly
ten-second check** that the nightly job ran:

```bash
systemctl list-timers glowstar-nightly
```

That single command protects the thing that matters most — grid freshness is the
largest measured driver of pricing error in this system.
