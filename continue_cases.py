"""Continue paired near-target states with refined grids and energy perturbations."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json, time
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import resample, find_peaks
from scipy.fft import fft
from config import ROOT
from batch_engine import BatchEngine

if os.environ.get("MAP_GPU") == "1":
    from gpu_engine import BatchEngine
from batch_engine import aligned_residual


def run(group, index=52, rounds=6000, source_prefix=""):
    source = f"{source_prefix}group_{group:02d}.mat"
    m = loadmat(ROOT / source, simplify_cells=True)
    assert (
        m["completed"][index] == m["trace"].shape[0]
    ), "Resume only a completed screening state"
    c = m["c"]
    c["passive_step_m"] = 0.25
    dt = min(0.25, float(m["dt"]))
    n = round(1024 / dt)
    t = (np.arange(n) - n / 2) * dt
    factors = np.array([0.9, 1.0, 1.1])
    a = np.repeat(resample(m["a"][index], n, axis=-1)[None], 3, axis=0) * np.sqrt(
        factors[:, None, None]
    )
    original_energy = np.sum(abs(m["a"][index]) ** 2) * m["dt"]
    assert np.allclose(
        np.sum(abs(a) ** 2, axis=(1, 2)) * dt, original_energy * factors, rtol=1e-10
    )
    pop = np.repeat(m["pop"][index][None], 3, axis=0)
    q = np.repeat(m["q"][index], 3)
    oc = np.repeat(m["oc"][index], 3)
    pump = np.repeat(m["pumps"][index], 3)
    engine = BatchEngine(c, dt, n, pump, oc)
    active = np.arange(3)
    trace = np.full((rounds, 3, 10), np.nan)
    previous = None
    final = np.zeros_like(a)
    final_a = a.copy()
    finalpop = pop.copy()
    finalq = q.copy()
    completed = np.zeros(3, int)
    states = ["running"] * 3
    start = time.time()
    name = f"long_g{group:02d}_c{index:02d}"
    for k in range(rounds):
        a, out, pop, q, sa = engine.step(a, pop, q)
        p = np.sum(abs(out) ** 2, axis=1)
        energy = p.sum(axis=1) * dt
        spec = np.sum(abs(fft(out, axis=-1)) ** 2, axis=1)
        te = p[:, abs(t) > 0.45 * n * dt].sum(axis=1) / np.maximum(
            p.sum(axis=1), 1e-250
        )
        se = spec[:, abs(engine.e.w) > 0.9 * np.pi / dt].sum(axis=1) / np.maximum(
            spec.sum(axis=1), 1e-250
        )
        ir = (
            np.full(len(active), np.nan)
            if previous is None
            else aligned_residual(previous, out)[0]
        )
        peaks = [
            len(find_peaks(x, height=0.1 * x.max(), prominence=0.1 * x.max())[0])
            for x in p
        ]
        trace[k, active] = np.column_stack(
            [
                energy,
                peaks,
                ir,
                te,
                se,
                pop.mean(axis=1),
                engine.last_gap,
                sa[:, 1],
                sa[:, 2],
                p.max(axis=1),
            ]
        )
        final[active] = out
        final_a[active] = a
        finalpop[active] = pop
        finalq[active] = q
        completed[active] = k + 1
        previous = out.copy()
        if (k + 1) % 20 == 0:
            bad = (te > 1e-6) | (se > 1e-8) | (energy < 1e-80) | ~np.isfinite(energy)
            for j in np.flatnonzero(bad):
                states[active[j]] = (
                    "spectral_boundary"
                    if se[j] > 1e-8
                    else "time_boundary" if te[j] > 1e-6 else "decayed_or_nonfinite"
                )
            keep = ~bad
            active = active[keep]
            a = a[keep]
            pop = pop[keep]
            q = q[keep]
            previous = previous[keep]
            engine.pumps = engine.pumps[keep]
            engine.oc = engine.oc[keep]
        if (k + 1) % 200 == 0 or not len(active):
            progress = dict(
                round=k + 1,
                active=len(active),
                energy_pJ=energy.tolist(),
                elapsed=time.time() - start,
            )
            (ROOT / f"{name}_progress.json").write_text(json.dumps(progress))
            print(name, progress, flush=True)
        if not len(active):
            break
    for i in active:
        states[i] = "completed"
    savemat(
        ROOT / f"{name}.mat",
        dict(
            c=c,
            dt=dt,
            t=t,
            trace=trace,
            completed=completed,
            out=final,
            a=final_a,
            pop=finalpop,
            q=finalq,
            factors=factors,
            source_group=group,
            source_index=index,
            source_file=source,
            previous_rounds=int(m["completed"][index]),
        ),
        do_compression=True,
    )
    result = []
    for i in range(3):
        tr = trace[: completed[i], i]
        tail = tr[-500:]
        result.append(
            dict(
                group=group,
                index=index,
                factor=float(factors[i]),
                rounds=int(completed[i]),
                status=states[i],
                energy_pJ=float(tr[-1, 0]),
                energy_min_pJ=float(tail[:, 0].min()),
                energy_max_pJ=float(tail[:, 0].max()),
                energy_cv=float(tail[:, 0].std() / max(tail[:, 0].mean(), 1e-250)),
                single_tail=bool(np.all(tail[:, 1] == 1)),
                residual=float(np.nanmax(tail[:, 2])),
                inversion_gap=float(tr[-1, 6]),
            )
        )
    (ROOT / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf8")


if __name__ == "__main__":
    for group in [3, 8]:
        run(group)
