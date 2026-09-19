"""Fixed-capacity GPU screening; precise SciPy/adaptive certification stays separate."""

import json
import numpy as np
from scipy.signal import find_peaks
from config import config, map_initial, OC, PUMPS, GDD, TOPOLOGIES, ROOT


def scan(group, rounds=600, dt=0.5, n=2048, block=20):
    from gpu_engine import ResidentEngine

    if not 0 <= group < 10 or rounds < 1 or block < 1:
        raise ValueError("Invalid group/round/block")
    c = config(TOPOLOGIES[group // 5], GDD[group % 5])
    oc = np.repeat(OC, len(PUMPS))
    pump = np.tile(PUMPS, len(OC))
    state = map_initial(c, oc, n, dt)
    engine = ResidentEngine(c, dt, *state, pump, oc)
    traces = []
    for start in range(0, rounds, block):
        traces.append(engine.run(min(block, rounds - start)))
        # Fixed 64 slots: stable FFT shape; failed slots stay masked, not compacted.
    trace = np.concatenate(traces)
    final = engine.checkpoint()
    exact_peaks = np.full(len(oc), -1, dtype=int)
    for i in np.flatnonzero(~final["failed"]):
        power = np.sum(abs(final["out"][i]) ** 2, axis=0)
        exact_peaks[i] = len(
            find_peaks(power, height=0.1 * power.max(), prominence=0.1 * power.max())[0]
        )
    path = ROOT / f"resident_screen_group_{group:02d}.npz"
    np.savez_compressed(
        path,
        trace=trace,
        pump_W=pump,
        oc=oc,
        dt_ps=dt,
        final_exact_peak_count=exact_peaks,
        **final,
    )
    summary = dict(
        group=group,
        rounds=rounds,
        samples=n,
        dt_ps=dt,
        columns=[
            "intracavity_energy_pJ",
            "spectral_edge",
            "time_edge",
            "failed",
            "output_energy_pJ",
            "rough_peak_count",
            "inversion_gap",
        ],
        failed_cases=int(final["failed"].sum()),
        final_exact_single_peaks=int(np.count_nonzero(exact_peaks == 1)),
        scope="Fixed-grid screening only. Rough peaks are not SciPy prominence peaks.",
        full_cavity_single_pulse_certified=False,
        diagnostics="Boundary check at round-trip end; CPU adaptive replay required for candidates",
    )
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(scan(3))
