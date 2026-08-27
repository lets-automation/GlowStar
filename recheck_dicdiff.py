#!/usr/bin/env python3
"""Re-price the client's "dic diff" stones through the LIVE API and score them.

Run ON THE SERVER:  cd /opt/glowstar && .venv/bin/python recheck_dicdiff.py

Scores the SHIPPED path (the endpoint their CRM calls), not a local stand-in --
CLAUDE.md Trap 5. Compares three things: what the file said, what we say now,
and the desk's own price.
"""
import json, os, statistics, urllib.request

STONES = [
 {
  "stoneId": "BFP-218",
  "shape": "RBC",
  "weight": 0.3,
  "color": "D",
  "clarity": "VS2",
  "cut": "GD",
  "polish": "VG",
  "symmetry": "GD",
  "fluorescence": "NON",
  "desk": -54.0,
  "file_ai": -48.84
 },
 {
  "stoneId": "BFP-256",
  "shape": "RBC",
  "weight": 0.3,
  "color": "D",
  "clarity": "VS2",
  "cut": "GD",
  "polish": "VG",
  "symmetry": "VG",
  "fluorescence": "NON",
  "desk": -54.0,
  "file_ai": -48.84
 },
 {
  "stoneId": "BFP-248",
  "shape": "RBC",
  "weight": 0.3,
  "color": "E",
  "clarity": "VS2",
  "cut": "VG",
  "polish": "EX",
  "symmetry": "GD",
  "fluorescence": "NON",
  "desk": -50.0,
  "file_ai": -46.69
 },
 {
  "stoneId": "BFP-252",
  "shape": "RBC",
  "weight": 0.3,
  "color": "E",
  "clarity": "VS2",
  "cut": "VG",
  "polish": "VG",
  "symmetry": "GD",
  "fluorescence": "NON",
  "desk": -50.0,
  "file_ai": -46.69
 },
 {
  "stoneId": "BFP-253",
  "shape": "RBC",
  "weight": 0.3,
  "color": "F",
  "clarity": "SI1",
  "cut": "VG",
  "polish": "EX",
  "symmetry": "GD",
  "fluorescence": "NON",
  "desk": -53.0,
  "file_ai": -48.07
 },
 {
  "stoneId": "BFP-37",
  "shape": "RBC",
  "weight": 0.3,
  "color": "F",
  "clarity": "VVS2",
  "cut": "VG",
  "polish": "VG",
  "symmetry": "GD",
  "fluorescence": "NON",
  "desk": -53.0,
  "file_ai": -48.99
 },
 {
  "stoneId": "OW26-507",
  "shape": "RBC",
  "weight": 0.3,
  "color": "G",
  "clarity": "VS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -32.0,
  "file_ai": -35.14
 },
 {
  "stoneId": "OX26-525",
  "shape": "RBC",
  "weight": 0.3,
  "color": "G",
  "clarity": "VS1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -32.0,
  "file_ai": -35.41
 },
 {
  "stoneId": "PN26-375",
  "shape": "RBC",
  "weight": 0.3,
  "color": "G",
  "clarity": "VS1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -32.0,
  "file_ai": -35.41
 },
 {
  "stoneId": "PN26-92",
  "shape": "RBC",
  "weight": 0.31,
  "color": "G",
  "clarity": "VS1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -32.0,
  "file_ai": -35.33
 },
 {
  "stoneId": "BFP-255",
  "shape": "RBC",
  "weight": 0.32,
  "color": "D",
  "clarity": "SI1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -34.0,
  "file_ai": -39.63
 },
 {
  "stoneId": "BFP-265",
  "shape": "RBC",
  "weight": 0.32,
  "color": "D",
  "clarity": "VS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -35.0,
  "file_ai": -38.96
 },
 {
  "stoneId": "BFP-262",
  "shape": "RBC",
  "weight": 0.32,
  "color": "F",
  "clarity": "VS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -33.0,
  "file_ai": -36.24
 },
 {
  "stoneId": "BFP-279",
  "shape": "RBC",
  "weight": 0.32,
  "color": "I",
  "clarity": "SI1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -45.25,
  "file_ai": -40.41
 },
 {
  "stoneId": "OQ26-192",
  "shape": "RBC",
  "weight": 0.32,
  "color": "G",
  "clarity": "VVS1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -36.0,
  "file_ai": -40.13
 },
 {
  "stoneId": "PN26-492",
  "shape": "RBC",
  "weight": 0.32,
  "color": "D",
  "clarity": "VS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -34.0,
  "file_ai": -38.96
 },
 {
  "stoneId": "PN26-508",
  "shape": "RBC",
  "weight": 0.32,
  "color": "G",
  "clarity": "VVS1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -36.0,
  "file_ai": -40.13
 },
 {
  "stoneId": "BFI-199",
  "shape": "RBC",
  "weight": 0.33,
  "color": "D",
  "clarity": "VS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "FNT",
  "desk": -39.5,
  "file_ai": -42.93
 },
 {
  "stoneId": "OW26-57",
  "shape": "RBC",
  "weight": 0.33,
  "color": "G",
  "clarity": "VS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -33.0,
  "file_ai": -36.19
 },
 {
  "stoneId": "OX26-495",
  "shape": "RBC",
  "weight": 0.33,
  "color": "G",
  "clarity": "VS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -33.0,
  "file_ai": -36.19
 },
 {
  "stoneId": "PG26-20",
  "shape": "RBC",
  "weight": 0.33,
  "color": "G",
  "clarity": "VS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "FNT",
  "desk": -37.0,
  "file_ai": -41.25
 },
 {
  "stoneId": "PN26-164",
  "shape": "RBC",
  "weight": 0.33,
  "color": "F",
  "clarity": "VS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -33.0,
  "file_ai": -36.23
 },
 {
  "stoneId": "PG26-74",
  "shape": "RBC",
  "weight": 0.36,
  "color": "G",
  "clarity": "VS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "FNT",
  "desk": -40.0,
  "file_ai": -43.18
 },
 {
  "stoneId": "OR26-179",
  "shape": "RBC",
  "weight": 0.4,
  "color": "K",
  "clarity": "VVS1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -52.0,
  "file_ai": -47.82
 },
 {
  "stoneId": "OW26-277",
  "shape": "RBC",
  "weight": 0.4,
  "color": "I",
  "clarity": "VS2",
  "cut": "VG",
  "polish": "EX",
  "symmetry": "VG",
  "fluorescence": "NON",
  "desk": -51.75,
  "file_ai": -48.63
 },
 {
  "stoneId": "PA26-367",
  "shape": "RBC",
  "weight": 0.4,
  "color": "F",
  "clarity": "IF",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "VG",
  "fluorescence": "NON",
  "desk": -49.0,
  "file_ai": -54.09
 },
 {
  "stoneId": "PL26-123",
  "shape": "RBC",
  "weight": 0.4,
  "color": "E",
  "clarity": "VVS1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "MED",
  "desk": -53.0,
  "file_ai": -56.78
 },
 {
  "stoneId": "OR26-125",
  "shape": "RBC",
  "weight": 0.42,
  "color": "J",
  "clarity": "SI2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -49.0,
  "file_ai": -44.59
 },
 {
  "stoneId": "PR26-40",
  "shape": "RBC",
  "weight": 0.42,
  "color": "I",
  "clarity": "VVS1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "MED",
  "desk": -50.0,
  "file_ai": -54.15
 },
 {
  "stoneId": "PR26-138",
  "shape": "RBC",
  "weight": 0.42,
  "color": "K",
  "clarity": "VS1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "MED",
  "desk": -55.0,
  "file_ai": -50.78
 },
 {
  "stoneId": "BFM-49",
  "shape": "RBC",
  "weight": 0.52,
  "color": "G",
  "clarity": "VVS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "STG",
  "desk": -59.25,
  "file_ai": -53.72
 },
 {
  "stoneId": "OW26-117",
  "shape": "RBC",
  "weight": 0.53,
  "color": "F",
  "clarity": "VVS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -41.0,
  "file_ai": -44.71
 },
 {
  "stoneId": "OW26-59",
  "shape": "RBC",
  "weight": 0.53,
  "color": "H",
  "clarity": "SI1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -39.5,
  "file_ai": -42.71
 },
 {
  "stoneId": "PR26-61",
  "shape": "RBC",
  "weight": 0.6,
  "color": "I",
  "clarity": "VVS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "MED",
  "desk": -45.0,
  "file_ai": -52.68
 },
 {
  "stoneId": "OQ26-52",
  "shape": "RBC",
  "weight": 0.61,
  "color": "J",
  "clarity": "VS1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "MED",
  "desk": -51.0,
  "file_ai": -54.76
 },
 {
  "stoneId": "BFN-28",
  "shape": "RBC",
  "weight": 0.63,
  "color": "J",
  "clarity": "VS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "FNT",
  "desk": -45.0,
  "file_ai": -52.94
 },
 {
  "stoneId": "PN26-261",
  "shape": "RBC",
  "weight": 0.67,
  "color": "I",
  "clarity": "VS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -35.0,
  "file_ai": -44.03
 },
 {
  "stoneId": "BFQ-28",
  "shape": "RBC",
  "weight": 0.73,
  "color": "H",
  "clarity": "VVS1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "FNT",
  "desk": -45.0,
  "file_ai": -52.82
 },
 {
  "stoneId": "PR26-133",
  "shape": "RBC",
  "weight": 0.73,
  "color": "I",
  "clarity": "VVS1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "MED",
  "desk": -45.0,
  "file_ai": -56.03
 },
 {
  "stoneId": "PQ26-69",
  "shape": "RBC",
  "weight": 0.77,
  "color": "J",
  "clarity": "IF",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "MED",
  "desk": -49.0,
  "file_ai": -54.15
 },
 {
  "stoneId": "PQ26-86",
  "shape": "RBC",
  "weight": 0.8,
  "color": "J",
  "clarity": "VVS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "FNT",
  "desk": -44.0,
  "file_ai": -40.02
 },
 {
  "stoneId": "PI26-20",
  "shape": "RBC",
  "weight": 1.0,
  "color": "E",
  "clarity": "VS1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "MED",
  "desk": -59.25,
  "file_ai": -55.55
 },
 {
  "stoneId": "PM26-11",
  "shape": "RBC",
  "weight": 1.0,
  "color": "E",
  "clarity": "SI1",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "NON",
  "desk": -54.5,
  "file_ai": -48.85
 },
 {
  "stoneId": "BFS-15",
  "shape": "RBC",
  "weight": 1.03,
  "color": "D",
  "clarity": "VS2",
  "cut": "EX",
  "polish": "EX",
  "symmetry": "EX",
  "fluorescence": "FNT",
  "desk": -48.0,
  "file_ai": -52.22
 }
]

KEY = ''
for line in open('/opt/glowstar/.env', encoding='utf-8'):
    if line.startswith('GS_API_KEY='):
        KEY = line.split('=', 1)[1].strip()

payload = [{k: v for k, v in s.items() if k not in ('desk', 'file_ai')} for s in STONES]
req = urllib.request.Request('http://localhost:8000/frontoffice/price',
                             data=json.dumps(payload).encode(),
                             headers={'Content-Type': 'application/json', 'X-API-Key': KEY})
rows = json.loads(urllib.request.urlopen(req, timeout=120).read())

now, old, errs = [], [], []
for s, r in zip(STONES, rows):
    ai = r.get('AIDiscount')
    if ai is None:
        errs.append((s['stoneId'], str(r.get('Error'))[:60])); continue
    now.append((s['stoneId'], s, float(ai) - s['desk'], s['file_ai'] - s['desk'], float(ai)))

def stat(v):
    return (f"mean {statistics.mean(v):+.2f}  MAE {statistics.mean(abs(x) for x in v):.2f}  "
            f">=5pt {sum(1 for x in v if abs(x) >= 5) / len(v) * 100:.0f}%")

print(f'priced {len(now)}/{len(STONES)}')
if errs:
    print('ERRORS:', errs)
print()
print('variance vs the DESK price')
print('  file said :', stat([n[3] for n in now]))
print('  we say now:', stat([n[2] for n in now]))
print()
print('fluoro tiers (CLAUDE.md: cap ONLY I-M and Faint; NEVER D-E at Medium+)')
for label, keep in (
    ('D-E @ MED/STG ', lambda s: s['color'] in 'DE' and s['fluorescence'] in ('MED', 'STG')),
    ('I-M or FAINT  ', lambda s: s['color'] in ('I','J','K','L','M') or s['fluorescence'] == 'FNT'),
    ('NON           ', lambda s: s['fluorescence'] == 'NON'),
):
    sel = [n for n in now if keep(n[1])]
    if sel:
        print(f'  {label} n={len(sel):2d}  file {stat([n[3] for n in sel])}   now {stat([n[2] for n in sel])}')
print()
print('biggest gaps now:')
for sid, s, vnow, vfile, ai in sorted(now, key=lambda n: -abs(n[2]))[:8]:
    print(f"  {sid:10s} {s['weight']:.2f}ct {s['color']}/{s['clarity']:4s} {s['fluorescence']:4s}"
          f"  ours {ai:7.2f}  desk {s['desk']:7.2f}  gap {vnow:+6.2f}  (file was {vfile:+6.2f})")
