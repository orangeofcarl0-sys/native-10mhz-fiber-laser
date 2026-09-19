"""Full product grid; independent states with explicit numerical-failure masks."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys, time, json
import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
from scipy.io import savemat
from batch_engine import BatchEngine, aligned_residual
from config import ROOT, OC, PUMPS, GDD, TOPOLOGIES, config, map_initial


def scan(group, rounds=600, dt=0.5, n=2048, tag="", passive_step=0.5):
    topology = TOPOLOGIES[group // 5]
    gdd = GDD[group % 5]
    c = config(topology, gdd)
    c["passive_step_m"] = passive_step
    c["grid_dt_ps"] = dt
    c["grid_n"] = n
    prefix = f"{tag}group_{group:02d}"
    EngineClass = BatchEngine
    if os.environ.get("MAP_GPU") == "1":
        from gpu_engine import BatchEngine as EngineClass
    oc = np.repeat(OC, len(PUMPS))
    pump = np.tile(PUMPS, len(OC))
    count = len(oc)
    t = (np.arange(n) - n / 2) * dt
    a, pop, q = map_initial(c, oc, n, dt)
    initial_a = a.copy()
    initial_pop = pop.copy()
    engine = EngineClass(c, dt, n, pump, oc)
    # E, peak, FWHM, peaks, aligned residual, time edge, spectral edge,
    # CNT E, CNT peak, min CNT absorption, inversion gap, mean inversion.
    trace = np.full((rounds, count, 12), np.nan)
    final_a = np.zeros_like(a)
    final_out = np.zeros_like(a)
    final_pop = pop.copy()
    final_q = q.copy()
    reasons = ["running"] * count
    completed = np.zeros(count, int)
    active = np.arange(count)
    previous = None
    last_fields = []
    start = time.time()
    for k in range(rounds):
        a, out, pop, q, sa = engine.step(a, pop, q)
        power = np.sum(abs(out) ** 2, axis=1)
        energy = power.sum(axis=1) * dt
        spectrum = np.sum(abs(fft(out, axis=-1)) ** 2, axis=1)
        edge = power[:, abs(t) > 0.45 * n * dt].sum(axis=1) / np.maximum(
            power.sum(axis=1), 1e-250
        )
        spectral = spectrum[:, abs(engine.e.w) > 0.9 * np.pi / dt].sum(
            axis=1
        ) / np.maximum(spectrum.sum(axis=1), 1e-250)
        residual = (
            np.full(len(active), np.nan)
            if previous is None
            else aligned_residual(previous, out)[0]
        )
        peaks = []
        width = []
        for p in power:
            ids, _ = find_peaks(
                p, height=0.1 * p.max(), prominence=0.1 * p.max(), distance=1
            )
            peaks.append(len(ids))
            pk = int(np.argmax(p))
            l = pk
            r = pk
            while l > 0 and p[l] > 0.5 * p[pk]:
                l -= 1
            while r < n - 1 and p[r] > 0.5 * p[pk]:
                r += 1
            width.append((r - l) * dt)
        trace[k, active] = np.column_stack(
            [
                energy,
                power.max(axis=1),
                width,
                peaks,
                residual,
                edge,
                spectral,
                sa[:, 0],
                sa[:, 1],
                sa[:, 2],
                engine.last_gap,
                pop.mean(axis=1),
            ]
        )
        final_a[active] = a
        final_out[active] = out
        final_pop[active] = pop
        final_q[active] = q
        completed[active] = k + 1
        # Store the last 16 fields at fixed global case indices for diagnostic comparison.
        snapshot = np.full((count, 2, n), np.nan + 0j)
        snapshot[active] = out
        last_fields.append(snapshot)
        last_fields = last_fields[-16:]
        previous = out.copy()
        if (k + 1) % 20 == 0:
            bad_time = edge > 1e-6
            bad_spectrum = spectral > 1e-8
            dead = energy < 1e-80
            bad = bad_time | bad_spectrum | dead | ~np.isfinite(energy)
            for j in np.flatnonzero(bad):
                reasons[active[j]] = (
                    "spectral_boundary"
                    if bad_spectrum[j]
                    else "time_boundary" if bad_time[j] else "decayed_or_nonfinite"
                )
            keep = ~bad
            active = active[keep]
            a = a[keep]
            pop = pop[keep]
            q = q[keep]
            previous = previous[keep]
            engine.pumps = engine.pumps[keep]
            engine.oc = engine.oc[keep]
        if (k + 1) % 50 == 0 or not len(active):
            progress = dict(
                group=group,
                topology=topology,
                gdd=float(gdd),
                round=k + 1,
                active=len(active),
                elapsed=time.time() - start,
            )
            (ROOT / f"{prefix}_progress.json").write_text(json.dumps(progress))
            print(progress, flush=True)
        if not len(active):
            break
    for j in active:
        reasons[j] = "completed"
    rows = []
    for i in range(count):
        tr = trace[: completed[i], i]
        tail = tr[-100:]
        ir = tr[-64:, 4]
        mean = float(np.mean(tail[:, 0]))
        cv = float(np.std(tail[:, 0]) / max(mean, 1e-250))
        residual = float(np.nanmax(ir))
        single = bool(np.all(tr[-64:, 3] == 1))
        numerical = bool(
            reasons[i] == "completed"
            and max(tr[-64:, 5]) < 1e-6
            and max(tr[-64:, 6]) < 1e-8
        )
        in_energy = bool(np.min(tail[:, 0]) >= 100 and np.max(tail[:, 0]) <= 500)
        candidate = bool(
            numerical and single and in_energy and residual < 1e-3 and cv < 1e-4
        )
        rows.append(
            dict(
                id=f"g{group:02d}_c{i:02d}",
                group=group,
                index=i,
                topology=topology,
                oc=float(oc[i]),
                pump_mW=float(pump[i] * 1000),
                GDD_ps2=float(gdd),
                rounds=int(completed[i]),
                status=reasons[i],
                energy_pJ=float(tr[-1, 0]),
                energy_mean_pJ=mean,
                energy_min_pJ=float(np.min(tail[:, 0])),
                energy_max_pJ=float(np.max(tail[:, 0])),
                energy_cv=cv,
                peaks=int(tr[-1, 3]),
                single_tail=single,
                residual=residual,
                edge_max=float(np.max(tr[-64:, 5])),
                spectral_edge_max=float(np.max(tr[-64:, 6])),
                FWHM_ps=float(tr[-1, 2]),
                CNT_energy_pJ=float(tr[-1, 7]),
                CNT_peak_W=float(tr[-1, 8]),
                inversion_gap=float(tr[-1, 10]),
                inversion_drift=float(abs(tail[-1, 11] - tail[0, 11])),
                numerical_pass=numerical,
                energy_range_pass=in_energy,
                screen_candidate=candidate,
            )
        )
    savemat(
        ROOT / f"{prefix}.mat",
        dict(
            c=c,
            dt=dt,
            t=t,
            oc=oc,
            pumps=pump,
            rep=engine.e.rep,
            trace=trace,
            completed=completed,
            a=final_a,
            out=final_out,
            pop=final_pop,
            q=final_q,
            initial_a=initial_a,
            initial_pop=initial_pop,
            fields=np.stack(last_fields, axis=3),
        ),
        do_compression=True,
    )
    (ROOT / f"{prefix}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print("FINISHED", group, "cases", count, "seconds", time.time() - start, flush=True)


if __name__ == "__main__":
    scan(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 600)
