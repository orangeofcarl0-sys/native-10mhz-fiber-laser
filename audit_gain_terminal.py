"""Local terminal grid check kept separate from the original run manifest."""

import json
import numpy as np
from config import ROOT, config
from run_gain_relaxation import grid_probe

manifest = json.loads((ROOT / "manifest.json").read_text())
assert manifest["status"] != "running"
with np.load(ROOT / "checkpoint.npz") as m:
    state = [m[k].copy() for k in ["a", "pop", "q"]]
    dt = float(m["dt"])
result = grid_probe(config("CNT_OC", 0.2), *state, dt)
(ROOT / "terminal_grid_probe.json").write_text(
    json.dumps(result, indent=2), encoding="utf8"
)
print(result)
