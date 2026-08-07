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
if [[ "${GS_DATABASE_URL:-}" == postgres* ]]; then
  pg_dump "$GS_DATABASE_URL" | gzip > "$BACKUP_DIR/db_$STAMP.sql.gz"
else
  sqlite3 /opt/glowstar/data/glowstar.db ".backup '$BACKUP_DIR/db_$STAMP.db'"
fi

# 3. feedback JSONL (small, and the training pipeline still reads it)
cp /opt/glowstar/data/feedback/decisions.jsonl "$BACKUP_DIR/decisions_$STAMP.jsonl" 2>/dev/null || true

# keep 30 days locally; anything older should already be off-box
find "$BACKUP_DIR" -name '*.gz' -mtime +30 -delete
find "$BACKUP_DIR" -name '*.db' -mtime +30 -delete

# OFF-SERVER copy. A backup on the same disk is not a backup.
if [[ -n "${GS_BACKUP_AZURE_URL:-}" ]]; then
  az storage blob upload-batch -d backups -s "$BACKUP_DIR" --overwrite \
     --connection-string "$GS_BACKUP_AZURE_URL" >/dev/null
fi
echo "backup complete: $BACKUP_DIR ($STAMP)"
