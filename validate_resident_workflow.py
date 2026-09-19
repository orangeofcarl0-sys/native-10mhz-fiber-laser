"""End-to-end 10RT timing including diagnostics, upload and final download."""

import json, time
import numpy as np
from config import config, map_initial, ROOT
from gpu_engine import ResidentEngine
from gpu_setup import cp
from scan_resident import scan

rows = []
for n in (2048, 8192):
    dt = 1024 / n
    c = config("OC_CNT", 0.2)
    oc = np.full(64, 0.3)
    pump = np.full(64, 0.02)
    initial = map_initial(c, oc, n, dt)
    times = {"reference": [], "spectral": []}
    checkpoints = {}
    # Warm both pipelines before timing; construction/upload/download included below.
    for mode in times:
        cfg = dict(c, gpu_pipeline=mode)
        ResidentEngine(cfg, dt, *initial, pump, oc).run(1)
    for repeat in range(3):
        for mode in (list(times) if repeat % 2 == 0 else list(times)[::-1]):
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            engine = ResidentEngine(dict(c, gpu_pipeline=mode), dt, *initial, pump, oc)
            trace = engine.run(10)
            checkpoints[mode] = engine.checkpoint()
            times[mode].append(time.perf_counter() - start)
    a, b = checkpoints["reference"], checkpoints["spectral"]
    np.testing.assert_array_equal(a["failed"], b["failed"])
    error = float(np.linalg.norm(a["a"] - b["a"]) / np.linalg.norm(a["a"]))
    assert error < 1e-9, error
    rows.append(
        dict(
            samples=n,
            batch=64,
            rounds=10,
            seconds=times,
            seconds_median={k: float(np.median(v)) for k, v in times.items()},
            field_relative_error=error,
        )
    )
    print(rows[-1], flush=True)
smoke = scan(3, rounds=40, block=20)
(ROOT / "resident_workflow_validation.json").write_text(
    json.dumps(dict(rows=rows, screen_smoke=smoke), indent=2)
)
print(smoke)
