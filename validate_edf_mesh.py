"""Orthogonal EDF-only mesh study: time/passive physics are held fixed."""

import json
from pathlib import Path
import numpy as np
from config import config, ROOT
from batch_engine import BatchEngine
from adaptive_solver import remap_population


def main():
    results = []
    for label in ["near_candidate", "boundary_snapshot", "multipulse_snapshot"]:
        fixture = np.load(Path(__file__).parent / "tests/fixtures" / f"{label}.npz")
        a = fixture["a"][None]
        pop = fixture["pop"][None]
        dt = float(fixture["dt"])
        outputs = []
        for step in [0.1, 0.05, 0.025]:
            c = config("OC_CNT", 0.2)
            c["max_step_m"] = step
            e = BatchEngine(c, dt, a.shape[-1], np.array([0.02]), np.array([0.3]))
            pin = np.array([0.02 * 10 ** (-c["pump_path_dB"] / 10)])
            n = remap_population(pop, e.e.ops[1][0])
            out, pump, n = e.fiber(a.copy(), 1, pin, n)
            outputs.append((out, pump, n))
        ref = outputs[-1]
        for step, (out, pump, n) in zip([0.1, 0.05, 0.025], outputs):
            results.append(
                dict(
                    case=label,
                    edf_step_m=step,
                    dt_ps=dt,
                    field_error_vs_0025=float(
                        np.linalg.norm(out - ref[0]) / np.linalg.norm(ref[0])
                    ),
                    energy_pJ=float(np.sum(abs(out) ** 2) * dt),
                    energy_relative_error=float(
                        abs(np.sum(abs(out) ** 2) / np.sum(abs(ref[0]) ** 2) - 1)
                    ),
                    remaining_pump_W=float(pump[0]),
                    population_error_vs_0025=float(
                        np.max(abs(remap_population(n, ref[2].shape[-1]) - ref[2]))
                    ),
                )
            )
    report = dict(
        scope="one EDF pass only; archived boundary input is diagnostic, not physically certified",
        rows=results,
    )
    (ROOT / "edf_mesh_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
