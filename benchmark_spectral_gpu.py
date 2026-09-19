"""Matched FP64 CUDA-event benchmark; no speedup claims from transform counts."""

import json, time
import numpy as np
from config import config, map_initial, ROOT
from gpu_setup import cp
from gpu_batch_core import BatchEngine as Reference, absorber
from gpu_kernels import absorber_prefix
from spectral_engine import SpectralEngine


def timed_suite(functions, repeats=7):
    for fun in functions.values():
        for _ in range(2):
            fun()
    cp.cuda.Stream.null.synchronize()
    samples = {key: [] for key in functions}
    walls = {key: [] for key in functions}
    rng = np.random.default_rng(721)
    for _ in range(repeats):
        for key in rng.permutation(list(functions)):
            start, end = cp.cuda.Event(), cp.cuda.Event()
            t = time.perf_counter()
            start.record()
            functions[key]()
            end.record()
            end.synchronize()
            samples[key].append(cp.cuda.get_elapsed_time(start, end))
            walls[key].append(1000 * (time.perf_counter() - t))
    return {
        key: dict(
            cuda_ms_median=float(np.median(values)),
            wall_ms_median=float(np.median(walls[key])),
            cuda_ms_samples=values,
        )
        for key, values in samples.items()
    }


records = []
for n in (2048, 8192):
    for topology in ("OC_CNT", "CNT_OC"):
        c = config(topology, 0.2)
        dt = 1024 / n
        oc = np.full(64, 0.3)
        pumps = np.full(64, 0.02)
        initial = tuple(cp.asarray(x) for x in map_initial(c, oc, n=n, dt=dt))
        engines = {
            "reference": Reference(c, dt, n, pumps, oc),
            "spectral_serial_cnt": SpectralEngine(
                c, dt, n, pumps, oc, gpu=True, fused_cnt=False, fused_edf=False
            ),
            "spectral_prefix_cnt": SpectralEngine(
                c, dt, n, pumps, oc, gpu=True, fused_edf=False
            ),
            "spectral_fused": SpectralEngine(c, dt, n, pumps, oc, gpu=True),
        }
        outputs = {
            key: engine.step(*(x.copy() for x in initial))
            for key, engine in engines.items()
        }
        errors = {}
        for key, result in outputs.items():
            errors[key] = float(
                cp.linalg.norm(result[0] - outputs["reference"][0])
                / cp.linalg.norm(outputs["reference"][0])
            )
            assert errors[key] < 1e-10, (key, errors[key])
            for i in (2, 3, 4):
                cp.testing.assert_allclose(
                    result[i], outputs["reference"][i], rtol=1e-10, atol=1e-12
                )
        timings = timed_suite(
            {
                key: (lambda e=engine: e.step(*(x.copy() for x in initial)))
                for key, engine in engines.items()
            }
        )
        cnt = timed_suite(
            {
                key: (lambda fun=fun: fun(initial[0], initial[2], c, dt))
                for key, fun in [("serial", absorber), ("prefix", absorber_prefix)]
            }
        )
        k = sum(x[0] for x in engines["spectral_prefix_cnt"].ops)
        edf = engines["spectral_prefix_cnt"].ops[1][0]
        records.append(
            dict(
                n=n,
                batch=64,
                topology=topology,
                dt_ps=dt,
                errors=errors,
                timings=timings,
                cnt=cnt,
                transforms_reference=2 * (k - edf) + 10 + 4 * edf,
                transforms_spectral=2 * k + 4,
            )
        )
        print(
            topology,
            n,
            {k: round(v["cuda_ms_median"], 3) for k, v in timings.items()},
            flush=True,
        )
result = dict(
    device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    cupy=cp.__version__,
    baseline="801fd03 FP64 GPU core, resident device inputs; identical input each timed round",
    methodology="CUDA events, 2 warmups, 7 interleaved randomized repeats, no host field copies; includes input device copies",
    nsight_available=False,
    records=records,
)
(ROOT / "spectral_gpu_benchmark.json").write_text(json.dumps(result, indent=2))
print("saved spectral_gpu_benchmark.json")
