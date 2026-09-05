"""Workstream B — Inventory Intelligence.

Consumes the Workstream-A pricing engine; never changes pricing math.

  survival.py   censored + left-truncated dataset, Kaplan-Meier with intervals
  velocity.py   per-stone expected days-to-sell (discrete-time hazard) and the
                own-velocity / market-depth pair that MOU 5.2 forbids merging
  bifurcate.py  the five MOU classes + ageing buckets
  reprice.py    velocity-adjusted repricing, bounded by the SHIPPED price band
  chart.py      dashboard JSON (MOU 5.1 - endpoints, not a screen)
"""
