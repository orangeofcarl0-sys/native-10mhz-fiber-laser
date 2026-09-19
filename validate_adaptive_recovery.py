"""Analytic near-band-edge pulse: successful rollback vs already-fine start."""

import json
import numpy as np
from config import config, map_initial, ROOT
from adaptive_solver import AdaptiveSolver, regrid

c = config("OC_CNT", 0.2)
c["max_step_m"] = 0.05
a, pop, q = map_initial(c, [0.3], 2048, 0.5)
t = (np.arange(2048) - 1024) * 0.5
a *= np.sqrt(80 / np.max(np.sum(abs(a) ** 2, axis=1)))
a *= np.exp(2j * np.pi * 0.8 * t)
solver = AdaptiveSolver(c, 0.02, 0.3, 0.5, 2048)
out = solver.step(a, pop, q)
fine = a.copy()
dt = 0.5
while dt > solver.dt:
    fine, dt = regrid(fine, dt, True, False)
while fine.shape[-1] < solver.n:
    fine, dt = regrid(fine, dt, False, True)
reference = AdaptiveSolver(solver.c, 0.02, 0.3, dt, fine.shape[-1])
from adaptive_solver import remap_population

expected = reference.step(fine, remap_population(pop, reference.engine.e.ops[1][0]), q)
error = float(np.linalg.norm(out[1] - expected[1]) / np.linalg.norm(expected[1]))
assert solver.events and any(x["action"] == "rollback_refine" for x in solver.events)
assert error < 1e-8, error
record = dict(
    scope="synthetic high-peak near-Nyquist stress test, not a cavity solution",
    output_field_relative_error=error,
    initial_dt_ps=0.5,
    final_dt_ps=solver.dt,
    accepted_rounds=solver.round,
    events=solver.events,
)
(ROOT / "adaptive_recovery_validation.json").write_text(
    json.dumps(record, indent=2), encoding="utf8"
)
print(json.dumps(record, indent=2))
