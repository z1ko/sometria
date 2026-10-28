"""Reduce knn_probe.py's printed table to the metrics.json every other runner leaves.

    python scripts/knn_reduce.py <knn.txt> <metrics.json>

knn_probe.py prints a table and writes nothing; its runners keep that stdout verbatim as
knn.txt and call this, so skip checks and the results generators need no special case for
kNN. Shared by scripts/knn_matrix.sh, goodnight_smpl.sh and goodnight_arms.sh.
"""

import json
import sys
from pathlib import Path

log, out = Path(sys.argv[1]), Path(sys.argv[2])
names, rows, source = None, {}, None
for line in log.read_text().splitlines():
    fields = line.split()
    if not fields:
        continue
    if fields[0] == "source":
        names = fields[1:]
    elif names and len(fields) > 2 and fields[1].startswith("k="):
        source = fields[0]
        rows[int(fields[1][2:])] = dict(zip(names, (float(value) for value in fields[2:])))

if not rows:
    raise SystemExit(f"no kNN table in {log}")

# Headline is the best k by macro mAP, with the whole sweep kept beside it -- the same
# shape probe_convex_mae.py reports its weight-decay sweep in, and read the same way: the
# sweep is nearly free because the distances are computed once, so reporting only the
# argmax would throw away the only evidence that k mattered at all.
best = max(rows, key=lambda k: rows[k]["macro_map"])
out.write_text(json.dumps({"metrics": rows[best], "k": best, "sweep": rows, "source": source}, indent=2))
