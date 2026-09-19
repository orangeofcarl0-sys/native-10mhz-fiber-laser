"""Compare resident GPU and host-interface GPU from identical FP64 states."""

import json
import time
import numpy as np
from config import config, map_initial, ROOT
from gpu_engine import BatchEngine, ResidentEngine
from gpu_setup import cp

records = []
for topology in ("OC_CNT", "CNT_OC"):
    c = config(topology, 0.2)
    oc = np.full(8, 0.3)
    pumps = np.full(8, 0.02)
    initial = map_initial(c, oc)
    times = {"host": [], "resident": []}
    for repeat in range(3):
        host = BatchEngine(c, 0.5, 2048, pumps, oc)
        resident = ResidentEngine(c, 0.5, *initial, pumps, oc)
        a, p, q = (x.copy() for x in initial)
        host.step(a.copy(), p.copy(), q.copy())  # warm kernels/plans
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            a, out, p, q, sa = host.step(a, p, q)
        times["host"].append(time.perf_counter() - start)
        start = time.perf_counter()
        trace = resident.run(10)
        checkpoint = resident.checkpoint()
        times["resident"].append(time.perf_counter() - start)
        assert not checkpoint["failed"].any(), trace[-1]
        error = float(np.linalg.norm(checkpoint["a"] - a) / np.linalg.norm(a))
        assert error < 1e-10
        np.testing.assert_allclose(checkpoint["pop"], p, atol=1e-13)
    records.append(
        dict(
            topology=topology,
            field_relative_error=error,
            seconds_median={k: float(np.median(v)) for k, v in times.items()},
            cases=8,
            rounds=10,
            samples=2048,
            note="Resident timing includes boundary diagnostics and final checkpoint",
        )
    )
# A nonfinite state must be flagged; no silent successful output.
bad = [x.copy() for x in initial]
bad[0][0, 0, 0] = np.nan
resident = ResidentEngine(c, 0.5, *bad, pumps, oc)
assert resident.run(1)[0, 0, 3] == 1
(ROOT / "resident_validation.json").write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
