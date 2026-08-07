#!/usr/bin/env bash
# One-shot provisioning for a fresh Ubuntu 24.04 LTS server.
#
# 24.04, NOT 22.04 — verified in a container, not assumed:
#   22.04 `apt install python3.11` yields Python 3.11.0rc1, a RELEASE CANDIDATE,
#   and PostgreSQL 14. 24.04 ships Python 3.12.3 and PostgreSQL 16 as its stable
#   defaults, and is supported to 2029 (22.04 ends 2027). Shipping a release
#   candidate interpreter to production is not a risk worth taking to save a
#   version number.
# Run as root on the new VM:  bash deploy/install.sh
#
# Everything below is decided in advance ON PURPOSE. Deployment day should be
# "run this, paste the credentials, done" — not a design session.
set -euo pipefail

# Non-interactive apt. Without this, `tzdata` (and occasionally postgresql) opens
# a full-screen menu asking for a geographic area and WAITS FOR INPUT — on a
# server being provisioned over SSH that reads as "the install has hung".
export DEBIAN_FRONTEND=noninteractive

APP_USER=glowstar
APP_DIR=/opt/glowstar

echo "==> system packages"
# `python3` (24.04's stable 3.12.3), NOT `python3.11` — 24.04 has no python3.11
# package at all, and on 22.04 that name resolves to a release candidate.
# `sudo` and `gzip` are listed explicitly: this script uses both, and a minimal
# Ubuntu cloud/container image does not always ship them. Without sudo the
# script dies at the first `sudo -u postgres` — found by running it on a clean
# machine, not by reading it.
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip \
    postgresql postgresql-contrib nginx certbot python3-certbot-nginx \
    sqlite3 git curl sudo gzip tzdata

echo "==> timezone"
# Azure (and most cloud) Ubuntu images default to UTC. systemd's OnCalendar uses
# LOCAL time, so a 02:30 timer on a UTC box fires at 08:00 IST — in the middle of
# the client's working day, competing with live pricing requests, and before the
# grid has settled. `backup.sh` also stamps filenames with `date +%F`, which
# would roll over at 05:30 IST and mislabel a night's backup.
#
# The client's data is IST: grid history timestamps carry +05:30.
#
# `timedatectl` needs systemd-timedated running, which is true on a real VM but
# NOT inside a build container — and with `set -e` a failure there would abort
# the whole install. So try it, and fall back to setting the zone files directly.
TZ_WANT="${GS_TIMEZONE:-Asia/Kolkata}"
if ! timedatectl set-timezone "$TZ_WANT" 2>/dev/null; then
  # The fallback must not "succeed" by creating a DANGLING symlink — ln -sf is
  # perfectly happy to point at a file that does not exist, and then `date`
  # silently keeps reporting UTC. Verify the zone data is actually there.
  [ -f "/usr/share/zoneinfo/$TZ_WANT" ] || {
    echo "  ERROR: /usr/share/zoneinfo/$TZ_WANT missing (is tzdata installed?)"; exit 1; }
  ln -sf "/usr/share/zoneinfo/$TZ_WANT" /etc/localtime
  echo "$TZ_WANT" > /etc/timezone
fi
echo "  timezone: $(date '+%Z %z')  ($TZ_WANT)"
# The nightly timer fires on LOCAL time; if this is not IST the job runs in the
# middle of the client's working day. Fail now rather than discover it at 08:00.
[ "$(date '+%z')" = "+0530" ] || {
  echo "  ERROR: timezone did not take effect — timer would fire at the wrong hour"; exit 1; }

echo "==> service account (no shell: nothing logs in as the app)"
id -u "$APP_USER" &>/dev/null || useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"
mkdir -p "$APP_DIR" /var/backups/glowstar
chown -R "$APP_USER:$APP_USER" "$APP_DIR" /var/backups/glowstar

echo "==> database"
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='glowstar'" | grep -q 1 || \
  sudo -u postgres psql -c "CREATE USER glowstar WITH PASSWORD '${GS_DB_PASSWORD:?set GS_DB_PASSWORD}';"
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='glowstar'" | grep -q 1 || \
  sudo -u postgres psql -c "CREATE DATABASE glowstar OWNER glowstar;"

echo "==> python environment"
sudo -u "$APP_USER" python3 -m venv "$APP_DIR/.venv"
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -q --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

echo "==> preflight: the Rapaport list must be on disk BEFORE the API starts"
# The Rap list is STATIC — there is no live feed, Glow Star sends the CSV — so
# it ships with the application payload rather than being downloaded. If it is
# missing the service still boots and /health still says 200, and then every
# single price request fails with FileNotFoundError. Fail here instead, loudly,
# where somebody is watching. (Found by running this on a clean machine: the API
# came up fine and then 500'd on the first quote.)
missing=0
for f in CSV2_ROUND_8_4.csv CSV2_PEAR_8_4.csv; do
  if [ ! -s "$APP_DIR/$f" ]; then echo "  MISSING: $APP_DIR/$f"; missing=1; fi
done
if [ "$missing" = "1" ]; then
  echo
  echo "  The Rapaport price list is not deployed. Copy the current CSVs into"
  echo "  $APP_DIR and re-run. This sheet is licensed and has no live feed: it"
  echo "  must come from Glow Star, never scraped or reconstructed. A stale sheet"
  echo "  is felt as 'the dollar prices are wrong'."
  exit 1
fi
echo "  ok: Rap list present"

echo "==> services"
cp "$APP_DIR/deploy/glowstar-api.service" /etc/systemd/system/
cp "$APP_DIR/deploy/glowstar-nightly.service" /etc/systemd/system/
cp "$APP_DIR/deploy/glowstar-nightly.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now glowstar-api glowstar-nightly.timer

echo
echo "Remaining, by hand (each needs a human decision or a secret):"
echo "  1. put credentials in $APP_DIR/.env   (see .env.example)"
echo "  2. nginx + TLS:  certbot --nginx -d <your-domain>"
echo "  3. first data load: sudo -u $APP_USER $APP_DIR/.venv/bin/python -m glowstar.market.grid_history"
echo "  4. verify:      curl localhost:8000/health"
echo "  5. verify a REAL PRICE, not just health — health returns 200 even when"
echo "     pricing is broken:"
echo "     curl -s -X POST localhost:8000/price -H \"X-API-Key: \$GS_API_KEY\" \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"Shape_full\":\"ROUND\",\"Weight\":1.01,\"Color\":\"G\",\"Clarity\":\"VS1\"}'"
