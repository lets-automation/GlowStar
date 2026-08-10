# Maintenance Runbook — How to Change Anything on the Server

**For:** Lets Automation (internal)
**Server:** Contabo Cloud VDS S · Ubuntu 24.04 · Navi Mumbai · `217.217.248.111`
**Deployed:** 2026-08-10

This is the "how do I actually do X" document. Every procedure is a numbered
sequence you can follow without thinking about it.

---

## 0. The map — what lives where

| Thing | Location |
|---|---|
| All application code | `/opt/glowstar/glowstar/` |
| Deployment scripts | `/opt/glowstar/deploy/` |
| Credentials | `/opt/glowstar/.env` (mode 600, owned by `glowstar`) |
| Python environment | `/opt/glowstar/.venv/` |
| Live inventory cache | `/opt/glowstar/records.json` (rebuilt nightly) |
| **Grid history (irreplaceable)** | `/opt/glowstar/data/master_grid/history.json` |
| Trained models | `/opt/glowstar/artifacts/models/` |
| Rapaport sheets | `/opt/glowstar/CSV2_ROUND_8_4.csv`, `CSV2_PEAR_8_4.csv` |
| Backups | `/var/backups/glowstar/` |
| Service definitions | `/etc/systemd/system/glowstar-*.service`, `.timer` |

**Two services:**

| Name | What it is | When it runs |
|---|---|---|
| `glowstar-api` | The pricing API | Always. Restarts on crash, starts on boot. |
| `glowstar-nightly` | Data pull → retrain → backup | 02:30 IST daily (+ up to 5 min jitter) |

**Connect:** `ssh root@217.217.248.111`

---

## 1. How the deployment was done (for reference / rebuilding)

If the server is ever lost, this is the whole sequence.

1. **Order** Contabo Cloud VDS S, region Asia (India), image **Ubuntu 24.04**.
2. **Harden:** `passwd`, `apt-get update && apt-get upgrade -y`, then
   `ufw allow 22,80,443/tcp` and `ufw --force enable`. Reboot.
3. **Build the bundle** on the dev machine (section 4) and copy it up:
   `scp deploy-bundle.tar.gz root@<IP>:/root/`
4. **Extract:** `mkdir -p /opt/glowstar && tar xzf /root/deploy-bundle.tar.gz -C /opt/glowstar`
5. **Credentials:** copy `.env` up, then `chmod 600 /opt/glowstar/.env`.
   **Strip Windows line endings** — see section 7, this has bitten us.
6. **Install:** `cd /opt/glowstar && export GS_DB_PASSWORD='…' && bash deploy/install.sh`
7. **Verify:** health, a real price, and the timer (section 6).
8. **Open the API port** (interim, until HTTPS): `ufw allow 8000/tcp`

The installer is idempotent — running it twice is safe and creates no duplicates.

---

## 2. How to update the CODE

### 2a. Small change — one or two files

From the **dev machine**:

```bash
scp glowstar/models/engine.py root@217.217.248.111:/opt/glowstar/glowstar/models/engine.py
```

On the **server**:

```bash
chown glowstar:glowstar /opt/glowstar/glowstar/models/engine.py
systemctl restart glowstar-api
```

> ⚠️ **ALWAYS verify the file actually landed before restarting.** A failed `scp`
> is silent, and we lost a deployment cycle to exactly this — the file looked
> copied, the old code kept running, and the same error reappeared:
>
> ```bash
> ls -l /opt/glowstar/glowstar/models/engine.py     # timestamp must be NOW
> grep -c "some_string_from_your_change" /opt/glowstar/glowstar/models/engine.py
> ```

### 2b. Bigger change — rebuild and redeploy the bundle

On the **dev machine**, always rebuild the bundle after changing code, or the
next deployment silently ships the old version:

```bash
CUR=$(python -c "import json;print(json.load(open('artifacts/models/current.json'))['version'])")
rm -f deploy-bundle.tar.gz
tar czf deploy-bundle.tar.gz --exclude='__pycache__' --exclude='*.pyc' \
  glowstar deploy tests requirements.txt pyproject.toml .env.example README.md \
  CSV2_ROUND_8_4.csv CSV2_PEAR_8_4.csv records.json \
  data/master_grid/history.json data/master_grid/current.json data/rap_versions \
  artifacts/market_segments.json artifacts/bgm_discounts.json artifacts/uni_codebook.json \
  artifacts/models/current.json "artifacts/models/$CUR"
```

Then on the **server**:

```bash
systemctl stop glowstar-api
cp -r /opt/glowstar/data/master_grid /root/grid-safety-copy   # never lose this
tar xzf /root/deploy-bundle.tar.gz -C /opt/glowstar
chown -R glowstar:glowstar /opt/glowstar
systemctl start glowstar-api
```

**Always run the tests before deploying:** `python -m pytest -q` on the dev
machine. Everything must pass.

### 2c. If a change breaks something — roll back the MODEL

Models are versioned. To go back to a previous one:

```bash
ls /opt/glowstar/artifacts/models/            # list versions
cat /opt/glowstar/artifacts/models/current.json
# edit current.json to an earlier version, then:
systemctl restart glowstar-api
curl -s localhost:8000/health                 # confirm the version changed
```

---

## 3. How to change CREDENTIALS or SETTINGS (`.env`)

```bash
nano /opt/glowstar/.env
grep -c $'\r' /opt/glowstar/.env      # MUST print 0 — see section 7
systemctl restart glowstar-api
curl -s localhost:8000/health
```

**Both services read `.env` at start**, so a restart is required for any change.

### Rotating the API key

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
nano /opt/glowstar/.env               # replace GS_API_KEY
systemctl restart glowstar-api
```

Then send the new key to the client securely. Old key stops working immediately.

### Rotating the database password

Two places, and they **must match exactly**:

```bash
sudo -u postgres psql -c "ALTER USER glowstar WITH PASSWORD 'NEW';"
nano /opt/glowstar/.env               # update the password inside GS_DATABASE_URL
systemctl restart glowstar-api
curl -s localhost:8000/health         # 'store' must show counts, not an error
```

### Useful settings

| Variable | Effect |
|---|---|
| `GS_RATE_LIMIT` | Requests/min per caller. `0` disables. Default 120. |
| `GS_USE_FEEDBACK` | **Leave unset/0.** Turning it on costs ~+0.9 MAE and makes the gate reject every candidate. |
| `GS_BACKUP_RCLONE_REMOTE` | Off-site backup destination |
| `GS_BACKTEST_SPLIT` | Evaluation split date (`2026-06-01`) |

---

## 4. How to run things by hand

| Task | Command |
|---|---|
| Full nightly job now | `sudo systemctl start glowstar-nightly` |
| Watch it run | `journalctl -u glowstar-nightly -f` (**Ctrl+C** to exit) |
| Data pull only | `sudo -u glowstar /opt/glowstar/.venv/bin/python -m glowstar.ingestion.run_snapshot` |
| Retrain only | `sudo -u glowstar /opt/glowstar/.venv/bin/python -m glowstar.training.retrain` |
| Backup only | `sudo -u glowstar bash -c 'set -a; . /opt/glowstar/.env; set +a; /opt/glowstar/deploy/backup.sh'` |
| System status | `sudo -u glowstar /opt/glowstar/.venv/bin/python -m glowstar.status` |

Anything run as the app must run **as the `glowstar` user**, or it writes files
that the service then cannot read.

The nightly job is safe to run twice and safe to run during the day.

---

## 5. How to add HTTPS (when the domain exists)

```bash
sed -i 's/YOUR_DOMAIN/pricing.example.com/g' /opt/glowstar/deploy/nginx.conf
cp /opt/glowstar/deploy/nginx.conf /etc/nginx/sites-available/glowstar
ln -sf /etc/nginx/sites-available/glowstar /etc/nginx/sites-enabled/glowstar
rm -f /etc/nginx/sites-enabled/default
nginx -t                                   # must say "successful"
certbot --nginx -d pricing.example.com
systemctl reload nginx
```

Then close the temporary plain-HTTP port and tighten the app to localhost:

```bash
ufw delete allow 8000/tcp
nano /etc/systemd/system/glowstar-api.service   # --host 0.0.0.0 -> 127.0.0.1
systemctl daemon-reload && systemctl restart glowstar-api
```

**Leave port 80 open** — the certificate renews through an inbound check on it.
Close it and everything works for ~60 days, then stops.

---

## 6. Health checks

```bash
systemctl status glowstar-api --no-pager        # must be: active (running)
curl -s localhost:8000/health                   # status, model version, data age
systemctl list-timers glowstar-nightly          # NEXT must be tomorrow 02:30 IST
ls -lh /var/backups/glowstar                    # today's files present?
df -h /                                         # disk
```

**`/health` returning 200 does NOT mean pricing works.** Always ask for a real
price — that distinction has caught real bugs here:

```bash
KEY=$(grep '^GS_API_KEY=' /opt/glowstar/.env | cut -d= -f2-)
curl -s -X POST localhost:8000/price -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"Shape_full":"ROUND","Weight":1.01,"Color":"G","Clarity":"VS1"}'
```

`"status":"degraded"` usually means the data is stale — check `records_age_hours`
and run the nightly job.

---

## 7. Traps that have actually bitten us here

**Windows line endings in `.env`.** Editing on Windows and copying up puts an
invisible `\r` on every line. The API key then contains a carriage return, the
HTTP header is malformed, and requests fail with no error body at all. After any
Windows-side edit:

```bash
sed -i 's/\r$//' /opt/glowstar/.env
grep -c $'\r' /opt/glowstar/.env      # must be 0
```

**A silent `scp`.** A failed copy leaves the old file in place and the same bug
"reappears". Always check the timestamp and grep for your change.

**Never delete `/opt/glowstar/data/master_grid/history.json`.** It is the
day-by-day record of the grid; the source API only serves recent dates, so a lost
day is lost permanently, and the grid model depends on it.

**Never turn on `GS_USE_FEEDBACK`** without redoing the calibration work. It
costs ~+0.9 MAE and freezes the nightly retrain by making the gate reject
everything.

**The server runs on IST.** It was UTC out of the box; the installer forces
`Asia/Kolkata` and aborts if that fails, because the timer fires on local time.

---

## 8. Emergency — the server is gone

1. Order a new Contabo VDS, Ubuntu 24.04, Mumbai.
2. Follow section 1.
3. Restore the grid history from backup **before** starting the services:
   ```bash
   gunzip -c grid_history_YYYY-MM-DD.json.gz > /opt/glowstar/data/master_grid/history.json
   ```
4. Restore the database:
   ```bash
   gunzip -c db_YYYY-MM-DD.sql.gz | sudo -u postgres psql glowstar
   ```
5. Run the nightly job once to refresh everything, then verify (section 6).

Estimated time: about an hour, most of it `pip install`.

**This only works if off-site backups exist.** That is the single most important
thing to keep working.
