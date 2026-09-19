"""Local time/space refinement check on the actual near-target states."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample
from batch_engine import BatchEngine, aligned_residual
from config import ROOT

results = []
for g, idx in [(2, 41), (3, 52), (8, 52)]:
    m = loadmat(ROOT / f"group_{g:02d}.mat", simplify_cells=True)
    outputs = []
    for dt, step in [(0.25, 0.25), (0.125, 0.125)]:
        c = m["c"].copy()
        c["passive_step_m"] = step
        c["max_step_m"] = 0.1
        n = round(1024 / dt)
        a = resample(m["a"][idx], n, axis=-1)[None]
        pop = m["pop"][idx][None].copy()
        q = np.array([m["q"][idx]])
        engine = BatchEngine(
            c, dt, n, np.array([m["pumps"][idx]]), np.array([m["oc"][idx]])
        )
        for k in range(20):
            a, out, pop, q, sa = engine.step(a, pop, q)
        outputs.append((out, dt))
    low = outputs[0][0]
    high = resample(outputs[1][0], low.shape[-1], axis=-1)
    e = [float(np.sum(abs(out) ** 2) * dt) for out, dt in outputs]
    results.append(
        dict(
            group=g,
            index=idx,
            rounds=20,
            low_energy_pJ=e[0],
            high_energy_pJ=e[1],
            relative_energy_difference=abs(e[1] - e[0]) / e[1],
            aligned_intensity_difference=float(aligned_residual(low, high)[0][0]),
        )
    )
    (ROOT / "resolution_check.json").write_text(
        json.dumps(results, indent=2), encoding="utf8"
    )
    print(results[-1], flush=True)
