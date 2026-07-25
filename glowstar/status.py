"""Print the engine's CURRENT state, measured — not asserted.

Exists so that no document (CLAUDE.md, a brief, a handoff prompt) ever has to
hardcode a model version, an MAE, a row count or a test count. Those rot within
days; this command cannot. Any doc that wants a number should say "run
`python -m glowstar.status`" instead of quoting one.

    python -m glowstar.status
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .config import SETTINGS, REPO_ROOT


def _age(ts) -> str:
    if ts is None:
        return "?"
    try:
        d = datetime.fromtimestamp(ts, tz=timezone.utc)
        hrs = (datetime.now(timezone.utc) - d).total_seconds() / 3600
        return f"{d:%Y-%m-%d %H:%M}Z ({hrs:.0f}h ago)"
    except Exception:
        return "?"


def main() -> None:
    print("=" * 66)
    print("GLOWSTAR ENGINE STATUS")
    print("=" * 66)

    # --- model in production ---
    from .models import registry
    try:
        eng, card = registry.load_current()
    except Exception as e:
        eng, card = None, None
        print(f"model         : FAILED TO LOAD ({type(e).__name__}: {e})")
    if card:
        print(f"model         : {card.get('version')}   trained {card.get('trained_at')}")
        print(f"  out-of-time : MAE={card.get('test_mae')}  "
              f"within5={card.get('test_within5')}  "
              f"coverage={card.get('test_coverage')}")
        print(f"  trained on  : {card.get('n_train', '?'):,} rows"
              if isinstance(card.get("n_train"), int)
              else f"  trained on  : {card.get('n_train', '?')} rows")
        print(f"  promoted    : {card.get('promoted')}  ({card.get('notes', '')})")
        print(f"  versions    : {len(registry.list_versions())} in registry")

    # --- fluoro caps actually baked into the live model ---
    caps = getattr(eng, "_fluor_caps", None) if eng is not None else None
    if caps:
        cells = ", ".join(f"{b}/{t}={v:+.1f}" for (b, t), v in sorted(caps.items()))
        print(f"  fluoro caps : {cells}")
        bad = [k for k in caps if k[0] == "D-E" and k[1] != "Faint"]
        if bad:
            print("  *** WARNING: D-E is capped at Medium+ — this is the bug that "
                  "put stones 5-10pts off the desk's price. See CLAUDE.md. ***")
    elif eng is not None:
        print("  fluoro caps : NONE (raw model penalty)")
    defer = getattr(eng, "_defer_shapes", None) if eng is not None else None
    if defer:
        print(f"  deferred    : {sorted(defer)} (competence guard -> segment median)")

    # --- data ---
    rec = REPO_ROOT / "records.json"
    if rec.exists():
        try:
            data = json.load(open(rec, encoding="utf-8"))
            from collections import Counter
            c = Counter(x.get("Status") for x in data)
            print(f"records.json  : {len(data):,} rows  {dict(c)}")
            print(f"  refreshed   : {_age(rec.stat().st_mtime)}")
        except Exception as e:
            print(f"records.json  : UNREADABLE ({type(e).__name__})")
    else:
        print("records.json  : MISSING (run the retrain to rebuild from live)")

    # --- feedback (recorded vs actually used) ---
    from .feedback import store as fbstore
    try:
        fb = fbstore.load_all()
        from collections import Counter
        d = Counter(r.get("decision") for r in fb)
        use_train = os.environ.get("GS_USE_FEEDBACK", "0") != "0"
        print(f"feedback      : {len(fb)} recorded {dict(d)}")
        print(f"  in training : {'ON' if use_train else 'OFF'} (GS_USE_FEEDBACK)")
        print("  in pricing  : OFF by default "
              "(price_and_report(use_feedback=False))")
        if use_train:
            print("  *** WARNING: feedback ON in training. Measured cost +0.93 MAE; "
                  "the gate will reject every candidate and the nightly retrain "
                  "will silently freeze. See CLAUDE.md. ***")
    except Exception as e:
        print(f"feedback      : unreadable ({type(e).__name__})")

    # --- config that changes pricing ---
    print(f"backtest split: {SETTINGS.backtest_split_date}")
    print(f"rapaport list : STATIC ({SETTINGS.rap_csv if hasattr(SETTINGS, 'rap_csv') else 'April CSV'}) "
          "— affects $/ct, not the discount")
    print("=" * 66)
    print("Numbers above are MEASURED now. Do not copy them into any document —")
    print("point at this command instead.")


if __name__ == "__main__":
    main()
