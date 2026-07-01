"""Cut-aware market segmentation (a VG stone must match VG market, not the EX blend)."""

from __future__ import annotations

from glowstar.market.segments import cut_tier, segment_keys


def test_cut_tier_buckets():
    assert cut_tier("3EX") == "EX"
    assert cut_tier("EX") == "EX"
    assert cut_tier("EX-VG") == "EX"      # leading token = cut grade
    assert cut_tier("VG-EX") == "VG"
    assert cut_tier("VG") == "VG"
    assert cut_tier("GD") == "LOW"
    assert cut_tier("") == "EX"           # unknown -> top (don't over-discount)
    assert cut_tier(None) == "EX"


def test_segment_keys_cut_aware_prefix_then_backoff():
    base = segment_keys("Round", 0.9, "I", "SI2")
    sb = base[0][1]
    withcut = segment_keys("Round", 0.9, "I", "SI2", "VG-EX")
    # 4 CUT-AWARE levels (most -> least specific), THEN the cut-blind base. A VG
    # stone always matches a VG market before any cut-blind level.
    assert withcut[:4] == [("Round", sb, "I", "SI2", "VG"), ("Round", sb, "I", "VG"),
                           ("Round", sb, "VG"), ("Round", "VG")]
    assert withcut[4:] == base
    # backward compatible: no cut -> unchanged base keys.
    assert segment_keys("Round", 0.9, "I", "SI2") == base
