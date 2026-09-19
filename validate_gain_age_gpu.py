"""Multi-cavity GPU rate diagnostics and accumulated conditional age regression."""

import json
import numpy as np
from config import config, map_initial, ROOT
from batch_engine import BatchEngine as CPU
from gpu_engine import BatchEngine as GPU
from gain_age import advance

rows = []
for topology in ["OC_CNT", "CNT_OC"]:
    c = config(topology, 0.2)
    oc = np.array([0.2, 0.5, 0.8])
    pump = np.array([0.01, 0.03, 0.05])
    a, p, q = map_initial(c, oc, n=512)
    b, pb, qb = a.copy(), p.copy(), q.copy()
    cpu = CPU(c, 0.5, 512, pump, oc)
    gpu = GPU(c, 0.5, 512, pump, oc)
    age = np.zeros_like(p)
    ageb = age.copy()
    for _ in range(8):
        a, out, p, q, sa = cpu.step(a, p, q)
        b, ob, pb, qb, sb = gpu.step(b, pb, qb)
        age = advance(age, cpu.rates_b, 1 / cpu.e.rep)
        ageb = advance(ageb, gpu.rates_b, 1 / gpu.e.rep)
        np.testing.assert_allclose(gpu.rates_b, cpu.rates_b, rtol=1e-10)
        np.testing.assert_allclose(ageb, age, rtol=1e-10)
        np.testing.assert_allclose(b, a, rtol=1e-8, atol=1e-10)
    rows.append(
        dict(
            topology=topology,
            cases=3,
            rounds=8,
            rates_max_relative=float(np.max(abs(gpu.rates_b / cpu.rates_b - 1))),
            age_max_relative=float(np.max(abs(ageb / age - 1))),
        )
    )
(ROOT / "gpu_gain_age_validation.json").write_text(
    json.dumps(rows, indent=2), encoding="utf8"
)
print(rows)
