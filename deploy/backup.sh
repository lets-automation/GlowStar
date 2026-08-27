#!/usr/bin/env bash
# Nightly backup of the things that CANNOT be rebuilt.
#
# data/master_grid/history.json is the record of what every price cell read on
# every day. The API only serves a short window of history, so once a day passes
# unrecorded it is gone for good — and without it the grid model cannot be
# honestly validated (today's grid explaining a past sale is leakage). Losing it
# is the one unrecoverable failure in this system.
set -euo pipefail
BACKUP_DIR="${GS_BACKUP_DIR:-/var/backups/glowstar}"
STAMP=$(date +%F)
mkdir -p "$BACKUP_DIR"

# 1. the irreplaceable grid history
gzip -c /opt/glowstar/data/master_grid/history.json \
  > "$BACKUP_DIR/grid_history_$STAMP.json.gz"

# 2. the database (quotes, decisions, scores)
#
# GS_DATABASE_URL is a SQLAlchemy URL and carries a DRIVER suffix:
#     postgresql+psycopg://glowstar:pw@localhost/glowstar
# `pg_dump` speaks libpq, which has no idea what "+psycopg" means — it treats
# the whole string as a database NAME and fails with
#     FATAL: database "postgresql+psycopg://..." does not exist
# so the nightly dump would fail EVERY night while the grid backup beside it
# succeeded. Strip the driver suffix to get a URL libpq understands.
# NEVER FALL BACK SILENTLY. This used to drop to a sqlite branch whenever
# GS_DATABASE_URL was unset. systemd sets it from EnvironmentFile, so the
# NIGHTLY run was fine — but a human running `bash deploy/backup.sh` from a root
# shell has no such variable, took the sqlite branch, and sqlite3 CREATED an
# empty database and "backed up" 4096 bytes of nothing. It then sat in the
# listing beside the real 536 KB dumps looking like a backup, on a day the
# nightly had already failed. A backup you believe in but do not have is worse
# than no backup, so an unset URL is now a hard error.
if [[ -z "${GS_DATABASE_URL:-}" ]]; then
  echo "ERROR: GS_DATABASE_URL is not set - refusing to write a fake backup." >&2
  echo "       systemd supplies it via EnvironmentFile. To run this by hand:" >&2
  echo "         sudo -u glowstar bash -c 'set -a; . /opt/glowstar/.env; set +a; bash /opt/glowstar/deploy/backup.sh'" >&2
  exit 1
fi

if [[ "${GS_DATABASE_URL}" == postgres* ]]; then
  PG_URL="$(printf '%s' "$GS_DATABASE_URL" | sed -E 's#^(postgresql|postgres)\+[a-z0-9_]+://#\1://#')"
  pg_dump "$PG_URL" | gzip > "$BACKUP_DIR/db_$STAMP.sql.gz"
  DB_FILE="$BACKUP_DIR/db_$STAMP.sql.gz"
else
  sqlite3 "${GS_DATABASE_URL#sqlite:///}" ".backup '$BACKUP_DIR/db_$STAMP.db'"
  DB_FILE="$BACKUP_DIR/db_$STAMP.db"
fi

# A dump can also fail UPWARD into an empty-but-present file (bad credentials, a
# killed pg_dump). Assert a plausible size rather than trusting exit status 0.
DB_BYTES=$(stat -c%s "$DB_FILE" 2>/dev/null || echo 0)
if (( DB_BYTES < 20000 )); then
  echo "ERROR: $DB_FILE is only $DB_BYTES bytes - that is not a real dump." >&2
  exit 1
fi

# 3. feedback JSONL (small, and the training pipeline still reads it)
cp /opt/glowstar/data/feedback/decisions.jsonl "$BACKUP_DIR/decisions_$STAMP.jsonl" 2>/dev/null || true

# keep 30 days locally; anything older should already be off-box
find "$BACKUP_DIR" -name '*.gz' -mtime +30 -delete
find "$BACKUP_DIR" -name '*.db' -mtime +30 -delete

# ---------------------------------------------------------------------------
# OFF-SERVER copy. A backup on the same disk is not a backup.
#
# Push-only by design: this server sends the files OUT to storage. Nothing ever
# connects inward to us, and no firewall port is opened anywhere.
#
# Only TODAY's files are sent. The previous version re-uploaded the whole 30-day
# directory every night — ~600 MB of pointless transfer for ~20 MB of new data.
#
# Prefer a destination at a DIFFERENT provider from the server itself. Backups
# that share a provider with the thing they are protecting fail together.
# ---------------------------------------------------------------------------
today_files=("$BACKUP_DIR"/*"$STAMP"*)
sent=""

if [[ -n "${GS_BACKUP_RCLONE_REMOTE:-}" ]]; then
  # Works with any S3-compatible or cloud store: Contabo Object Storage,
  # Backblaze B2, AWS S3, Azure Blob, Google Cloud Storage.
  # Configure once with `rclone config`, then set e.g.
  #   GS_BACKUP_RCLONE_REMOTE=b2:glowstar-backups
  rclone copy --no-traverse "${today_files[@]}" "$GS_BACKUP_RCLONE_REMOTE/" \
    || { echo "OFF-SITE BACKUP FAILED (rclone) — local copy kept" >&2; exit 1; }
  sent="$GS_BACKUP_RCLONE_REMOTE"

elif [[ -n "${GS_BACKUP_AZURE_URL:-}" ]]; then
  for f in "${today_files[@]}"; do
    az storage blob upload -c backups -f "$f" -n "$(basename "$f")" --overwrite \
       --connection-string "$GS_BACKUP_AZURE_URL" >/dev/null \
      || { echo "OFF-SITE BACKUP FAILED (azure) — local copy kept" >&2; exit 1; }
  done
  sent="azure:backups"
fi

if [[ -z "$sent" ]]; then
  # Loud, not silent. A backup that only exists on the machine being backed up
  # is the failure mode people discover on the day they need it.
  echo "WARNING: no off-site destination configured — set GS_BACKUP_RCLONE_REMOTE" >&2
  echo "         or GS_BACKUP_AZURE_URL. Backups are LOCAL ONLY." >&2
else
  echo "off-site copy sent to $sent"
fi

echo "backup complete: $BACKUP_DIR ($STAMP)"
