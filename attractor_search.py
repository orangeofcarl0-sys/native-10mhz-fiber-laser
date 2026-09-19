"""Pump-conditioned search, guarded gain acceleration, then direct validation.

Accelerated search counters are iterations, NEVER elapsed physical time.
No small-window run certifies absence of satellites in the full 100 ns cavity.
"""

from collections import deque
from copy import deepcopy
import json
import numpy as np
from scipy.signal import find_peaks
from scipy.io import savemat
from diagnostics import align
from shared_gain import integrated_stokes
from adaptive_solver import (
    AdaptiveSolver,
    NumericalLimit,
    pump_initial,
    remap_population,
)
from config import ROOT, config, map_initial


def accelerated_gain(pop, A, B, rep, max_change=0.002):
    """Damped exact frozen-rate relaxation; a search update, not a trajectory."""
    multiplier = min(1000.0, max(1.0, 0.1 * rep / float(B.max())))
    proposed = pop + (A / B - pop) * (-np.expm1(-B * multiplier / rep))
    damping = min(1.0, max_change / max(float(np.max(abs(proposed - pop))), 1e-100))
    return pop + damping * (proposed - pop), dict(
        nominal_round_multiplier=multiplier, damping=damping
    )


def stokes(a):
    x, y = a
    return np.array(
        [
            abs(x) ** 2 - abs(y) ** 2,
            2 * np.real(x * np.conj(y)),
            2 * np.imag(x * np.conj(y)),
        ]
    )


def field_distance(reference, current):
    """Remove only common time translation and global optical phase."""
    intensity, field, shift = align(reference, current)
    shifted = np.fft.ifft(
        np.fft.fft(current, axis=-1)
        * np.exp(-2j * np.pi * np.fft.fftfreq(current.shape[-1]) * shift),
        axis=-1,
    )
    polarization = float(
        np.linalg.norm(stokes(shifted) - stokes(reference))
        / max(np.linalg.norm(np.sum(abs(reference) ** 2, axis=0)), 1e-250)
    )
    return intensity, field, polarization


def classify(fields, populations, dt, relaxation_age=0.0, max_period=16):
    """Finite-record recurrence classes; never infer chaos from nonconvergence."""
    if len(fields) < 64:
        return dict(status="insufficient_history", period=None)
    f = list(fields)
    p = list(populations)
    energy = np.array([np.sum(abs(a) ** 2) * dt for a in f])
    peaks = np.array(
        [
            len(
                find_peaks(
                    np.sum(abs(a) ** 2, axis=0),
                    height=0.1 * np.sum(abs(a) ** 2, axis=0).max(),
                    prominence=0.1 * np.sum(abs(a) ** 2, axis=0).max(),
                )[0]
            )
            for a in f
        ]
    )
    metrics = dict(
        integrated_stokes_mean=np.mean(
            [integrated_stokes(a) for a in f], axis=0
        ).tolist(),
        integrated_stokes_span=np.ptp(
            [integrated_stokes(a) for a in f], axis=0
        ).tolist(),
        energy_mean_pJ=float(energy.mean()),
        energy_cv=float(energy.std() / max(energy.mean(), 1e-250)),
        energy_min_pJ=float(energy.min()),
        energy_max_pJ=float(energy.max()),
        peaks_max=int(peaks.max()),
        relaxation_age=float(relaxation_age),
    )
    if energy.max() < 1e-8:
        return dict(status="decayed", period=None, **metrics)
    for period in range(1, min(max_period, (len(f) - 1) // 4) + 1):
        errors = np.array(
            [field_distance(f[k - period], f[k]) for k in range(period, len(f))]
        )
        dn = np.array([p[k] - p[k - period] for k in range(period, len(p))])
        nrel = max(
            np.linalg.norm(p[k] - p[k - period]) / max(np.linalg.norm(p[k]), 1e-100)
            for k in range(period, len(p))
        )
        # Check long-lag longitudinal drift too; a spatial mean is not sufficient.
        drift = float(np.max(abs(p[-1] - p[-1 - 4 * period])))
        erel = float(
            np.max(
                abs(energy[period:] - energy[:-period])
                / np.maximum(energy[period:], 1e-250)
            )
        )
        cycles = (
            energy[-(len(energy) // period) * period :].reshape(-1, period).mean(axis=1)
        )
        cycle_cv = float(cycles.std() / max(cycles.mean(), 1e-250))
        if (
            errors[:, 1].max() < 1e-3
            and errors[:, 2].max() < 1e-3
            and erel < 1e-4
            and cycle_cv < 1e-4
            and np.max(abs(dn)) < 1e-5
            and nrel < 1e-4
            and drift < 1e-5
        ):
            single = bool(np.all(peaks == 1))
            target = bool(energy.min() >= 100 and energy.max() <= 500)
            return dict(
                status=(
                    "local_recurrence_validated"
                    if relaxation_age >= 5
                    else "provisional_recurrence"
                ),
                period=period,
                single_in_window=single,
                target_energy=target,
                intensity_residual=float(errors[:, 0].max()),
                field_residual=float(errors[:, 1].max()),
                stokes_residual=float(errors[:, 2].max()),
                population_relative=float(nrel),
                population_max_drift=drift,
                **metrics,
            )
    return dict(status="nonstationary_or_unresolved", period=None, **metrics)


def run_point(
    topology="OC_CNT",
    gdd=0.2,
    pump=0.02,
    oc=0.3,
    *,
    initialization="pump",
    seed=0,
    state=None,
    dt=0.5,
    n=2048,
    edf_step=0.05,
    options=None,
    search_blocks=10,
    block_rounds=32,
    direct_rounds=1000,
    name="adaptive_point",
):
    c = config(topology, gdd)
    supplied_state = state is not None
    c["max_step_m"] = edf_step
    solver = AdaptiveSolver(c, pump, oc, dt, n, options)
    if state is None:
        if initialization == "legacy":
            a, pop, q = map_initial(c, [oc], n, dt)
        elif initialization in ["pump", "noise"]:
            a, pop, q = pump_initial(
                c, pump, oc, n, dt, seed=seed, noisy=initialization == "noise"
            )
        else:
            raise ValueError(initialization)
    else:
        a, pop, q = (x.copy() for x in state)
        if a.shape[-1] != n:
            raise ValueError("Continuation must preserve the supplied time grid")
        pop = remap_population(pop, solver.engine.e.ops[1][0])
    trace = []
    jumps = []
    fields = deque(maxlen=80)
    populations = deque(maxlen=80)
    age = 0.0
    direct_done = 0
    status = "search_budget_exhausted"
    generation = solver.grid_generation

    def advance(frozen, phase):
        nonlocal a, pop, q, generation, age, direct_done
        a, out, pop, q, sa = solver.step(a, pop, q, frozen=frozen)
        if generation != solver.grid_generation:
            fields.clear()
            populations.clear()
            generation = solver.grid_generation
            if phase == "direct":
                age = 0.0  # Certification must age on the accepted final grid.
        e = float(np.sum(abs(out) ** 2) * solver.dt)
        tau = float(solver.engine.last_tau[0])
        if phase == "direct":
            age += 1 / (solver.engine.e.rep * tau)
            direct_done += 1
            fields.append(out[0].copy())
            populations.append(pop[0].copy())
        trace.append(
            dict(
                iteration=solver.round,
                phase=phase,
                energy_pJ=e,
                tau_s=tau,
                gain_gap=float(solver.engine.last_gap[0]),
                time_edge=float(solver.last_edges[0]),
                spectral_edge=float(solver.last_edges[1]),
                dt_ps=solver.dt,
                n=solver.n,
                inversion_mean=float(pop.mean()),
                z_steps=solver.engine.accepted_steps,
                z_rejects=solver.engine.rejected_steps,
            )
        )
        if solver.round % 16 == 0:
            (ROOT / f"{name}_progress.json").write_text(
                json.dumps(trace[-1]), encoding="utf8"
            )
        return out

    try:
        for block in range(search_blocks):
            fast = []
            rates_a = []
            rates_b = []
            for _ in range(block_rounds):
                before_generation = solver.grid_generation
                out = advance(True, "frozen_search")
                if solver.grid_generation != before_generation or (
                    fast and fast[-1].shape != out[0].shape
                ):
                    fast = []
                    rates_a = []
                    rates_b = []
                fast.append(out[0].copy())
                rates_a.append(solver.engine.rates_a.copy())
                rates_b.append(solver.engine.rates_b.copy())
                if len(fast) > 1:
                    ratio = np.sum(abs(fast[-1]) ** 2) / max(
                        np.sum(abs(fast[-2]) ** 2), 1e-250
                    )
                    if not 0.8 < ratio < 1.2:
                        break
            tail = fast[-8:]
            energies = np.array([np.sum(abs(x) ** 2) for x in tail])
            relaxed = (
                len(tail) >= 8
                and energies.mean() > 1e-8
                and energies.std() / energies.mean() < 0.01
                and max(
                    field_distance(tail[k - 1], tail[k])[1] for k in range(1, len(tail))
                )
                < 0.01
            )
            if relaxed:
                A = np.mean(rates_a[-8:], axis=0)
                B = np.mean(rates_b[-8:], axis=0)
                # Bound gain change and optical round-trip gain change; not physical time skipping.
                before = pop.copy()
                pop, jump = accelerated_gain(pop, A, B, solver.engine.e.rep)
                jumps.append(
                    dict(
                        block=block,
                        **jump,
                        max_population_change=float(np.max(abs(pop - before))),
                    )
                )
            else:
                # Frozen-field growth or oscillation invalidates the slow-manifold approximation.
                for _ in range(block_rounds):
                    advance(False, "dynamic_search")
        fields.clear()
        populations.clear()
        age = 0.0
        for _ in range(direct_rounds):
            out = advance(False, "direct")
            if direct_done % 64 == 0:
                result = classify(fields, populations, solver.dt, age)
                if result["status"] == "local_recurrence_validated":
                    break
        result = classify(fields, populations, solver.dt, age)
        status = result["status"]
    except NumericalLimit as exc:
        a, pop, q = solver.checkpoint if hasattr(solver, "checkpoint") else (a, pop, q)
        result = dict(status="numerically_unresolved", reason=str(exc))
        status = result["status"]
    result.update(
        topology=topology,
        gdd_ps2=gdd,
        pump_mW=pump * 1000,
        oc=oc,
        initialization="supplied_checkpoint" if supplied_state else initialization,
        search_iterations=sum(r["phase"] != "direct" for r in trace),
        direct_rounds=direct_done,
        direct_time_s=direct_done / solver.engine.e.rep,
        relaxation_age=age,
        final_dt_ps=solver.dt,
        final_n=solver.n,
        initial_edf_step_m=edf_step,
        edf_step_m=solver.c["max_step_m"],
        accelerated_updates=jumps,
        grid_events=solver.events,
        full_cavity_single_pulse_certified=False,
        numerical_options=solver.options,
    )
    (ROOT / f"{name}.json").write_text(
        json.dumps(dict(summary=result, trace=trace), indent=2), encoding="utf8"
    )
    savemat(
        ROOT / f"{name}.mat",
        dict(a=a, pop=pop, q=q, dt=solver.dt, c=solver.c, rounds=solver.round),
        do_compression=True,
    )
    return result, (a, pop, q), solver.dt


def continuation(topology, gdd, oc, values, *, axis="pump", pump_mW=20.0, **kwargs):
    """Both sweep directions start independently. Failed states are never reused."""
    results = []
    base_name = kwargs.pop("name", "continuation")
    initial_dt = kwargs.pop("dt", 0.5)
    initial_n = kwargs.pop("n", 2048)
    if axis not in ["pump", "gdd"]:
        raise ValueError("axis must be pump or gdd")
    for direction, sweep in [("up", values), ("down", list(reversed(values)))]:
        state = None
        dt = initial_dt
        n = initial_n
        for index, value in enumerate(sweep):
            r, next_state, next_dt = run_point(
                topology,
                gdd if axis == "pump" else float(value),
                (value if axis == "pump" else pump_mW) / 1000,
                oc,
                state=state,
                dt=dt,
                n=n,
                name=f"{base_name}_{direction}_{index:02d}",
                **kwargs,
            )
            r["direction"] = direction
            r["axis"] = axis
            # Unconverged warm starts may be explored, but are explicitly labelled, not attractors.
            r["seed_from_previous_nonconverged"] = (
                state is not None and previous != "local_recurrence_validated"
            )
            previous = r["status"]
            results.append(r)
            if r["status"] == "numerically_unresolved":
                state = None
            else:
                state = next_state
                dt = next_dt
                n = state[0].shape[-1]
    (ROOT / f"{base_name}.json").write_text(
        json.dumps(results, indent=2), encoding="utf8"
    )
    return results


def multi_seed(topology="OC_CNT", gdd=0.2, pump=0.02, oc=0.3, **kwargs):
    """Legacy equal state, pump-conditioned Gaussian and reproducible weak noise."""
    name = kwargs.pop("name", "multi_seed")
    rows = []
    for initialization, seed in [
        ("legacy", 0),
        ("pump", 0),
        ("noise", 1),
        ("noise", 2),
    ]:
        result, _, _ = run_point(
            topology,
            gdd,
            pump,
            oc,
            initialization=initialization,
            seed=seed,
            name=f"{name}_{initialization}_{seed}",
            **kwargs,
        )
        rows.append(result)
    (ROOT / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
    return rows


if __name__ == "__main__":
    result, _, _ = run_point()
    print(json.dumps(result, indent=2))
