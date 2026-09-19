"""Transient remote-satellite response of the archived, UNCONVERGED candidate."""

import json
from pathlib import Path
import numpy as np
from config import config, ROOT
from shared_gain import SharedGainEngine, satellite_windows, integrated_stokes

fixture = np.load(Path(__file__).parent / "tests/fixtures/near_candidate.npz")
c = config("OC_CNT", 0.2)
dt = float(fixture["dt"])
records = []
for eta in (0.0, 1e-6, 1e-4, 1e-2):
    a = satellite_windows(fixture["a"], eta)
    pop = fixture["pop"][None].copy()
    q = np.full(2, c["sa_modulation"])
    engine = SharedGainEngine(c, dt, a.shape[-1], 0.02, 0.3)
    trace = []
    for rt in range(1, 33):
        a, out, pop, q, sa = engine.step(a, pop, q)
        energy = np.sum(abs(out) ** 2, axis=(1, 2)) * dt
        power = np.sum(abs(a) ** 2, axis=1)
        spectrum = abs(np.fft.fft(a, axis=-1)) ** 2
        edge = float(
            spectrum[:, :, abs(np.fft.fftfreq(a.shape[-1])) > 0.45].sum()
            / max(spectrum.sum(), 1e-100)
        )
        time_edge = float(
            (power[:, : a.shape[-1] // 20].sum() + power[:, -a.shape[-1] // 20 :].sum())
            / max(power.sum(), 1e-100)
        )
        trace.append(
            dict(
                round=rt,
                primary_pJ=float(energy[0]),
                satellite_pJ=float(energy[1]),
                ratio=float(energy[1] / max(energy[0], 1e-100)),
                spectral_edge=edge,
                time_edge=time_edge,
                primary_stokes=integrated_stokes(a[0]).tolist(),
            )
        )
        if edge > 1e-8 or time_edge > 1e-6 or not np.isfinite(a).all():
            break
    records.append(
        dict(
            initial_ratio=eta,
            trace=trace,
            classification="transient_response_not_stability_certificate",
            reason="Base trajectory is not a converged fixed point or periodic orbit",
            full_cavity_single_pulse_certified=False,
        )
    )
(ROOT / "satellite_validation.json").write_text(json.dumps(records, indent=2))
print([(r["initial_ratio"], len(r["trace"]), r["trace"][-1]["ratio"]) for r in records])
