"""FP64 kernels, edge cases, evolving trajectories, and changing active batches."""

import json
import numpy as np
from config import config, map_initial, ROOT
from gpu_setup import cp
from batch_engine import absorber as cpu_absorber
from gpu_batch_core import BatchEngine as Reference, kerr
from gpu_kernels import absorber_prefix, kerr_inplace
from spectral_engine import SpectralEngine
from gpu_engine import BatchEngine, ResidentEngine

rng = np.random.default_rng(42)
cnt_errors = []
for n in (17, 257, 2048, 8192):
    for dt in (1e-5, 0.5, 10.0):
        c = config("OC_CNT", 0.2)
        a = (rng.normal(size=(3, 2, n)) + 1j * rng.normal(size=(3, 2, n))) * np.array(
            [0, 1, 100]
        )[:, None, None]
        q = np.array([0.0, 0.01, 0.05])
        reference = cpu_absorber(a, q.copy(), c, dt)
        result = absorber_prefix(cp.asarray(a), cp.asarray(q), c, dt)
        for x, y in zip(result, reference):
            cp.testing.assert_allclose(x, cp.asarray(y), rtol=2e-10, atol=1e-12)
        cnt_errors.append(
            float(
                cp.linalg.norm(result[0] - cp.asarray(reference[0]))
                / max(np.linalg.norm(reference[0]), 1e-100)
            )
        )
device = cp.asarray(a)
cp.testing.assert_allclose(
    kerr_inplace(device.copy(), 0.001), kerr(device, 0.001), rtol=1e-13, atol=1e-12
)
errors = []
for topology in ("OC_CNT", "CNT_OC"):
    for n in (2048, 8192):
        c = config(topology, 0.2)
        dt = 1024 / n
        c["segments"][:, 4] = 0.12
        c["segments"][:, 5] = np.linspace(0.1, 0.6, 6)
        c["segments"][:, 6] = 0.02
        oc = np.array([0.2, 0.5, 0.9])
        pump = np.array([0.005, 0.02, 0.15])
        initial = map_initial(c, oc, n, dt)
        initial[0][:, 1] = (0.2 + 0.1j) * initial[0][:, 0]
        old = Reference(c, dt, n, pump, oc)
        new = SpectralEngine(c, dt, n, pump, oc, gpu=True)
        a, p, q = (cp.asarray(x) for x in initial)
        b, pb, qb = (x.copy() for x in (a, p, q))
        for rt in range(20):
            a, out, p, q, sa = old.step(a, p, q)
            b, ob, pb, qb, sb = new.step(b, pb, qb)
            error = float(cp.linalg.norm(a - b) / cp.linalg.norm(a))
            errors.append(error)
            assert error < 1e-8, (topology, n, rt, error)
            cp.testing.assert_allclose(pb, p, rtol=1e-10, atol=1e-12)
            cp.testing.assert_allclose(ob, out, rtol=1e-7, atol=1e-9)
        print(topology, n, "20RT error", error, flush=True)
c = config("OC_CNT", 0.2)
initial = map_initial(c, [0.3, 0.4, 0.5])
wrapper = BatchEngine(c, 0.5, 2048, [0.02] * 3, [0.3, 0.4, 0.5])
a, out, p, q, sa = wrapper.step(*initial)
wrapper.pumps = np.array([0.02, 0.02])
wrapper.oc = np.array([0.3, 0.4])
wrapper.step(a[:2].copy(), p[:2].copy(), q[:2].copy())
assert wrapper.last_gap.shape == (2,)
frozen = dict(c, gain_mode="frozen")
frozen_engine = SpectralEngine(frozen, 0.5, 2048, [0.02] * 3, [0.3, 0.4, 0.5], gpu=True)
gpu_initial = tuple(cp.asarray(x) for x in initial)
frozen_result = frozen_engine.step(*(x.copy() for x in gpu_initial))
cp.testing.assert_array_equal(frozen_result[2], gpu_initial[1])
try:
    frozen_engine.step(
        gpu_initial[0].astype(cp.complex64), gpu_initial[1], gpu_initial[2]
    )
    raise AssertionError("FP32 must not enter FP64 kernels")
except ValueError:
    pass
# Resident map performs no per-step full-field host download, and flags bad CNT.
bad = dict(c)
bad["sa_nonsaturable"] = 1.1
resident = ResidentEngine(bad, 0.5, *initial, [0.02] * 3, [0.3, 0.4, 0.5])
assert resident.run(1)[0, :, 3].all()
np.testing.assert_array_equal(resident.checkpoint()["a"], initial[0])
result = dict(
    max_cnt_field_relative=max(cnt_errors),
    max_20round_field_relative=max(errors),
    cnt_cases=36,
    trajectory_rounds=80,
    rotated_birefringent_test=True,
    changing_batch_pass=True,
    invalid_transmission_rollback_pass=True,
)
(ROOT / "spectral_gpu_validation.json").write_text(json.dumps(result, indent=2))
print(result)
