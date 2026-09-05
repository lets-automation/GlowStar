#!/usr/bin/env bash
# GlowStar production audit pack — READ-ONLY. Run on the Contabo server:
#   sudo bash /opt/glowstar/scripts/server_audit.sh 2>&1 | tee /tmp/glowstar_audit_$(date +%F).txt
# Prints no secret values (only whether each variable is set).
set -u
cd /opt/glowstar || { echo "not at /opt/glowstar"; exit 1; }
PY=/opt/glowstar/.venv/bin/python
hdr() { printf '\n\n================ %s ================\n' "$1"; }

hdr "0. host"
date -u; uptime; hostnamectl 2>/dev/null | head -4
df -h /opt /var | tail -n +1; free -m
nproc; cat /proc/loadavg

hdr "1. code version on server vs git"
git -C /opt/glowstar rev-parse --short HEAD 2>/dev/null; git -C /opt/glowstar status --short 2>/dev/null | head -30
git -C /opt/glowstar log --format='%h %ad %s' --date=short -8 2>/dev/null
ls -la /opt/glowstar/glowstar/inventory 2>/dev/null | head -3 || echo "no glowstar/inventory on server (expected)"
md5sum glowstar/models/engine.py glowstar/market/grid_history.py glowstar/reference/normalize.py glowstar/service/frontoffice.py glowstar/service/app.py glowstar/service/pricing_service.py glowstar/training/retrain.py

hdr "2. services & timers"
systemctl is-active glowstar-api; systemctl is-enabled glowstar-api
systemctl status glowstar-api --no-pager -n 0 | head -12
systemctl list-timers glowstar-nightly --no-pager
systemctl status glowstar-nightly --no-pager -n 0 | head -12
echo "--- last 3 nightly runs (result lines) ---"
journalctl -u glowstar-nightly --no-pager -o short-iso 2>/dev/null | grep -E "Started|Deactivated|Failed|Main process exited|gate|promot|reject|MAE|Banked|Grid feature|Traceback|ERROR" | tail -60
echo "--- api restarts (ExecStartPost restart evidence) ---"
journalctl -u glowstar-api --no-pager -o short-iso 2>/dev/null | grep -E "Started|Stopped|Stopping" | tail -12
echo "--- api errors last 3 days ---"
journalctl -u glowstar-api --since "3 days ago" --no-pager 2>/dev/null | grep -E -i "error|traceback|exception|warning" | tail -30
echo "--- api worker uptime (model age in memory = process age) ---"
ps -o pid,etime,rss,cmd -C uvicorn 2>/dev/null; pgrep -af "uvicorn glowstar" | head

hdr "3. env (names only)"
for v in GS_API_KEY GS_ENV GS_DATABASE_URL GS_USE_FEEDBACK GS_USE_BGM GS_BACKUP_AZURE_URL GS_BACKUP_DIR CHANNEL_PARTNER_USER CHANNEL_PARTNER_PASS UNI_TOKEN DIAMANTO_USERNAME DIAMANTO_CLIENT_ID ANTHROPIC_API_KEY; do
  if grep -qE "^$v=" .env 2>/dev/null; then echo "$v: set"; else echo "$v: NOT SET"; fi
done
stat -c '%U:%G %a %n' .env

hdr "4. registry / model"
cat artifacts/models/current.json 2>/dev/null || cat artifacts/registry/current.json 2>/dev/null || find /opt/glowstar -name current.json -path "*model*" -exec sh -c 'echo {}; cat {}' \;
ls -lt artifacts/models 2>/dev/null | head -8
$PY - <<'EOF'
import json, glob, os, datetime as dt
from pathlib import Path
try:
    from glowstar.models import registry as R
    v = R.current_version(); print("current version:", v)
    for c in sorted(glob.glob(str(Path(R.REGISTRY_DIR if hasattr(R,'REGISTRY_DIR') else 'artifacts/models')/'*'/'card.json')))[-10:]:
        d = json.load(open(c)); print(os.path.basename(os.path.dirname(c)), {k: d.get(k) for k in ('trained_at','test_mae','metric_protocol','promoted','coverage','n_train','defer_shapes','test_n')})
except Exception as e:
    print("registry read failed:", e)
EOF

hdr "5. data freshness"
ls -la records.json data/master_grid/ 2>/dev/null
$PY - <<'EOF'
import json, datetime as dt, collections
h = json.load(open("data/master_grid/history.json"))
days = collections.Counter()
last = ""
for k, vs in h.items():
    for d, _ in vs:
        s = str(d)[:10]; days[s] += 1
        if s > last: last = s
print("grid history keys", len(h), " latest edit date:", last)
today = dt.date.today()
print("edits per day, last 30 days (0 = hole):")
for i in range(30, -1, -1):
    d = (today - dt.timedelta(days=i)).isoformat(); print(f"  {d} {days.get(d,0):7d}")
r = json.load(open("records.json")); recs = r if isinstance(r, list) else r.get("records")
import pandas as pd
R = pd.DataFrame(recs); print("records", len(R), R["Status"].value_counts().to_dict())
od = pd.to_datetime(R.loc[R.Status.eq("Sold"), "OrderDate"], errors="coerce"); print("last sale in records:", od.max())
EOF
ls -la data/snapshots/channel_partner 2>/dev/null | tail -4; ls -la data/snapshots/master_grid 2>/dev/null | tail -4
ls -la data/rap_versions 2>/dev/null; ls -la CSV2_*.csv 2>/dev/null

hdr "6. health + REAL prices (the /health check is not proof)"
KEY=$(grep -E '^GS_API_KEY=' .env | cut -d= -f2- | tr -d '"'"'"' ')
H='Content-Type: application/json'
curl -s localhost:8000/health | $PY -m json.tool | head -60
echo "--- same stone, 3 spellings (must agree) ---"
for body in \
 '{"Shape_full":"Round","Weight":1.01,"Color":"G","Clarity":"VS1","CPS":"3EX","Fluorescence":"NON","Lab":"GIA"}' \
 '{"Shape_full":"RBC","Weight":1.01,"Color":"g","Clarity":"vs1","CPS":"EX EX EX","Fluorescence":"None","Lab":"gia"}' \
 '{"Shape_full":"Round","Weight":1.01,"Color":"G","Clarity":"VS1","Fluorescence":"NON","Lab":"GIA"}' ; do
  echo "$body"; curl -s -X POST localhost:8000/price -H "$H" -H "X-API-Key: $KEY" -d "$body" | $PY -c 'import sys,json; d=json.load(sys.stdin); s=d.get("suggestion",d); print("  ->", {k:s.get(k) for k in ("suggested_discount","ci_discount_low","ci_discount_high","method","flags","feedback_correction_pts","comparable_count")})'
done
echo "--- same stone via /price 5 times (two workers: prices must be identical) ---"
for i in 1 2 3 4 5; do curl -s -X POST localhost:8000/price -H "$H" -H "X-API-Key: $KEY" -d '{"Shape_full":"Round","Weight":1.01,"Color":"G","Clarity":"VS1","CPS":"3EX","Fluorescence":"NON","Lab":"GIA"}' | $PY -c 'import sys,json; d=json.load(sys.stdin); s=d.get("suggestion",d); print("  ", s.get("suggested_discount"), s.get("feedback_correction_pts"))'; done
echo "--- pear via /price vs /frontoffice/price with no CPS ---"
curl -s -X POST localhost:8000/price -H "$H" -H "X-API-Key: $KEY" -d '{"Shape_full":"Pear","Weight":1.01,"Color":"G","Clarity":"VS1","Fluorescence":"NON","Lab":"GIA"}' | head -c 400; echo
curl -s -X POST localhost:8000/frontoffice/price -H "$H" -H "X-API-Key: $KEY" -d '[{"stoneId":"AUDIT-1","shape":"PB","weight":1.01,"color":"G","clarity":"VS1","fluorescence":"NON","lab":"GIA"}]' | head -c 600; echo
echo "--- same pear on FrontOffice as cps EX-EX vs polish/symmetry EX/EX (must agree) ---"
curl -s -X POST localhost:8000/frontoffice/price -H "$H" -H "X-API-Key: $KEY" -d '[{"stoneId":"AUDIT-2","shape":"PB","weight":1.01,"color":"G","clarity":"VS1","cps":"EX-EX","fluorescence":"NON","lab":"GIA"},{"stoneId":"AUDIT-3","shape":"PB","weight":1.01,"color":"G","clarity":"VS1","polish":"EX","symmetry":"EX","fluorescence":"NON","lab":"GIA"}]' | $PY -c 'import sys,json
for r in json.load(sys.stdin):
    print("  ", r.get("StoneId"), r.get("AIDiscount"), (r.get("Error") or "")[:80], r.get("Flags"))'
echo "--- batch with one bad row (should NOT fail the good one) ---"
curl -s -o /dev/null -w "HTTP %{http_code}\n" -X POST localhost:8000/price/batch -H "$H" -H "X-API-Key: $KEY" -d '[{"Shape_full":"Round","Weight":1.01,"Color":"G","Clarity":"VS1","CPS":"3EX","Fluorescence":"NON"},{"Shape_full":"Round","Weight":-1,"Color":"G","Clarity":"VS1","CPS":"3EX","Fluorescence":"NON"}]'
echo "--- is /docs public through nginx? ---"
curl -s -o /dev/null -w "docs via localhost: %{http_code}\n" localhost:8000/docs
curl -s -o /dev/null -w "no-key /price: %{http_code}\n" -X POST localhost:8000/price -H "$H" -d '{"Shape_full":"Round","Weight":1.01,"Color":"G","Clarity":"VS1","CPS":"3EX","Fluorescence":"NON"}'

hdr "7. feedback store (the /decision live-correction bug)"
ls -la data/feedback 2>/dev/null
$PY - <<'EOF'
try:
    from glowstar.feedback.store import load_all
    from glowstar.feedback.learning import build_corrections
    recs = load_all(); print("feedback records on disk:", len(recs))
    import collections; print(collections.Counter(r.get("decision") for r in recs))
    c = build_corrections(recs)
    print("correction cells that would be LIVE after any /decision call (min_support=3):", len(c))
    for k, v in list(c.items())[:15]: print("  ", k, v)
except Exception as e:
    print("feedback read failed:", e)
EOF

hdr "8. production DB counts"
$PY - <<'EOF'
from glowstar.store.db import get_engine, counts
print(counts())
import pandas as pd
e = get_engine()
print(pd.read_sql("select date_trunc('day', ts) d, count(*) n, min(model_version) mv_min, max(model_version) mv_max from quotes group by 1 order by 1 desc limit 21", e).to_string())
print(pd.read_sql("select source, count(*) from quotes group by 1", e).to_string())
print(pd.read_sql("select decision, count(*) from decisions group by 1", e).to_string())
print(pd.read_sql("select shape, count(*) from quotes group by 1 order by 2 desc limit 15", e).to_string())
EOF

hdr "9. backups"
ls -la ${GS_BACKUP_DIR:-/opt/glowstar/backups} 2>/dev/null | tail -8
grep -n "ExecStart\|ExecStartPost" /etc/systemd/system/glowstar-nightly.service

hdr "10. python stack"
$PY -c "import sklearn, numpy, pandas, fastapi, sqlalchemy; print('sklearn', sklearn.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'fastapi', fastapi.__version__, 'sqlalchemy', sqlalchemy.__version__)"
echo "DONE"
