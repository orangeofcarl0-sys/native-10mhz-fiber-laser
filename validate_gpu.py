import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import time, json
import numpy as np
from config import config, map_initial, ROOT
from batch_engine import BatchEngine as CPU
from gpu_engine import BatchEngine as GPU
from gpu_setup import cp

records = []
for topology in ["OC_CNT", "CNT_OC"]:
    c = config(topology, 0.2)
    pumps = np.array([0.005, 0.02, 0.15])
    oc = np.array([0.2, 0.5, 0.9])
    a, pop, q = map_initial(c, oc)
    ag = a.copy()
    pg = pop.copy()
    qg = q.copy()
    cpu = CPU(c, 0.5, 2048, pumps, oc)
    gpu = GPU(c, 0.5, 2048, pumps, oc)
    for j in range(10):
        a, out, pop, q, sa = cpu.step(a, pop, q)
        ag, og, pg, qg, sag = gpu.step(ag, pg, qg)
        error = np.linalg.norm(out - og) / np.linalg.norm(out)
        assert error < 1e-9, (topology, j, error)
        records.append(float(error))
c = config("OC_CNT", 0.2)
a, pop, q = map_initial(c, np.full(64, 0.8))
gpu = GPU(c, 0.5, 2048, np.full(64, 0.02), np.full(64, 0.8))
gpu.step(a, pop, q)
cp.cuda.Stream.null.synchronize()
start = time.perf_counter()
for j in range(10):
    a, out, pop, q, sa = gpu.step(a, pop, q)
seconds = time.perf_counter() - start
result = dict(max_relative_field_error=max(records), seconds_64_cases_10_rounds=seconds)
(ROOT / "gpu_validation.json").write_text(json.dumps(result, indent=2))
print(result)
