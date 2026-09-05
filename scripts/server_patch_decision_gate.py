"""Gate the live /decision correction refresh behind GS_USE_FEEDBACK on the SERVER.

The server's app.py is not byte-identical to the dev tree (it lacks the inventory
endpoints), so the file cannot be copied over. This applies the one change in place,
refusing to touch the file unless the exact original block is present once.

    sudo -u glowstar /opt/glowstar/.venv/bin/python /opt/glowstar/scripts/server_patch_decision_gate.py
    sudo systemctl restart glowstar-api

Idempotent: a second run reports "already patched" and exits 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

APP = Path("/opt/glowstar/glowstar/service/app.py")
if len(sys.argv) > 1:
    APP = Path(sys.argv[1])

OLD = """        # Online corrections refresh immediately so a repeated mistake shifts the
        # next quote without waiting for the nightly retrain.
        try:
            from ..feedback.learning import build_corrections
            from ..feedback import store as fbstore
            _get_service().engine.set_corrections(build_corrections(fbstore.load_all()))
        except Exception:
            log.exception("could not refresh online corrections (decision still stored)")
"""

NEW = """        # Online corrections refresh ONLY when feedback is explicitly enabled
        # (GS_USE_FEEDBACK, the same switch the trainer honours). CLAUDE.md Trap 3:
        # applying the desk's quotes as corrections measured +0.9 MAE, and with
        # --workers 2 this call armed only the worker that took the POST, so one
        # stone priced two ways until the nightly restart. The decision is still
        # stored either way; only the live price shift is gated.
        if os.environ.get("GS_USE_FEEDBACK", "0") != "0":
            try:
                from ..feedback.learning import build_corrections
                from ..feedback import store as fbstore
                _get_service().engine.set_corrections(build_corrections(fbstore.load_all()))
            except Exception:
                log.exception("could not refresh online corrections (decision still stored)")
"""

src = APP.read_text(encoding="utf-8")
if NEW in src:
    print("already patched:", APP)
    sys.exit(0)
n = src.count(OLD)
if n != 1:
    print(f"REFUSING: expected the original block exactly once in {APP}, found {n}. "
          "Open the file and apply the change by hand (see scripts/server_patch_decision_gate.py NEW).")
    sys.exit(2)
if "\nimport os\n" not in src:
    print("REFUSING: app.py has no `import os`; add it before patching.")
    sys.exit(2)
backup = APP.with_suffix(".py.bak-decision-gate")
backup.write_text(src, encoding="utf-8")
APP.write_text(src.replace(OLD, NEW), encoding="utf-8")
import py_compile
py_compile.compile(str(APP), doraise=True)
print("patched:", APP, " backup:", backup)
