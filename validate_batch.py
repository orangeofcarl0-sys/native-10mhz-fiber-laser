import time, json
from copy import deepcopy
import numpy as np
from config import config, map_initial, ROOT
from batch_engine import BatchEngine, aligned_residual
from python_engine import Engine
from diagnostics import align

records = []
for topology in ["OC_CNT", "CNT_OC"]:
    c = config(topology, 0.2)
    pumps = np.array([0.005, 0.02, 0.15])
    oc = np.array([0.2, 0.5, 0.9])
    engine = BatchEngine(c, 0.5, 2048, pumps, oc)
    a, pop, q = map_initial(c, oc)
    scalar = []
    for i in range(3):
        ci = deepcopy(c)
        ci["pump_ld_W"] = pumps[i]
        ci["output_power_fraction"] = oc[i]
        scalar.append([Engine(ci, 0.5, 2048), a[i].copy(), pop[i].copy(), q[i]])
    previous = None
    for step in range(10):
        a, out, pop, q, sa = engine.step(a, pop, q)
        for i, state in enumerate(scalar):
            e, ai, pi, qi = state
            ai, oi, pi, qi, ledger, gain, sai, local = e.step(ai, pi, qi)
            scalar[i] = [e, ai, pi, qi]
            error = np.linalg.norm(out[i] - oi) / np.linalg.norm(oi)
            assert error < 1e-10, (topology, i, error)
            assert np.max(abs(pop[i] - pi)) < 1e-12
            records.append(float(error))
        if previous is not None:
            ir, shift = aligned_residual(previous, out)
            for i in range(3):
                assert abs(ir[i] - align(previous[i], out[i])[0]) < 1e-10
        previous = out.copy()

c = config("OC_CNT", 0.2)
a, pop, q = map_initial(c, np.full(16, 0.8))
e = BatchEngine(c, 0.5, 2048, np.full(16, 0.02), np.full(16, 0.8))
start = time.perf_counter()
for step in range(20):
    a, out, pop, q, sa = e.step(a, pop, q)
seconds = time.perf_counter() - start
result = dict(
    max_field_relative=max(records),
    cases=6,
    rounds=10,
    seconds_16_cases_20_rounds=seconds,
    batch_seconds_per_case_round=seconds / 320,
)
(ROOT / "batch_validation.json").write_text(json.dumps(result, indent=2))
print(result)
