"""CPU reference solver with rollback/regrid and local SSFM error control.

Each instance follows one cavity. Search acceleration is deliberately separate
from real round-trip integration. Legacy fixed-grid engines remain available.
"""

from copy import deepcopy
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import resample
from scipy.integrate import solve_ivp
from batch_engine import BatchEngine, kerr
from config import map_initial

DEFAULTS = dict(
    z_rtol=1e-5,
    z_min_m=1e-5,
    nonlinear_phase_max=0.1,
    spectral_warning=1e-10,
    time_warning=1e-8,
    min_dt_ps=0.03125,
    max_samples=65536,
    max_window_ps=8192,
    max_retries=12,
    edf_rtol=1e-4,
    edf_population_atol=1e-5,
    min_edf_step_m=0.003125,
)


class NumericalLimit(RuntimeError):
    """Unresolved with current resources; never a physical rejection."""


class EDFRefinement(RuntimeError):
    """Internal request to replay a full round on a finer inversion mesh."""


class BoundaryRefinement(RuntimeError):
    def __init__(self, edges):
        self.edges = np.asarray(edges)


def health(a, dt):
    power = np.sum(abs(a) ** 2, axis=(0, 1))
    spectrum = np.sum(abs(fft(a, axis=-1)) ** 2, axis=(0, 1))
    n = a.shape[-1]
    t = (np.arange(n) - n / 2) * dt
    return (
        float(power[abs(t) > 0.45 * n * dt].sum() / max(power.sum(), 1e-250)),
        float(
            spectrum[abs(np.fft.fftfreq(n)) > 0.45].sum() / max(spectrum.sum(), 1e-250)
        ),
    )


def recenter(a, dt):
    """Neutral integer translation; accept only if edge occupation decreases.

    A periodic shift preserves all satellite pulses, not just a cropped peak.
    Solver limits its window to < one round trip, with a recovered CNT gap.
    """
    peak = int(np.argmax(np.sum(abs(a) ** 2, axis=(0, 1))))
    shift = a.shape[-1] // 2 - peak
    centered = np.roll(a, shift, axis=-1)
    if health(centered, dt)[0] < health(a, dt)[0]:
        return centered, shift * dt
    return a, 0.0


def regrid(a, dt, finer=False, wider=False):
    """Band-limited interpolation at fixed window, then symmetric zero padding."""
    before = np.sum(abs(a) ** 2) * dt
    if finer:
        a = resample(a, 2 * a.shape[-1], axis=-1)
        dt /= 2
    if wider:
        n = a.shape[-1]
        a = np.pad(a, ((0, 0), (0, 0), (n // 2, n - n // 2)))
    assert np.isclose(np.sum(abs(a) ** 2) * dt, before, rtol=1e-11, atol=1e-100)
    return a, dt


def remap_population(pop, cells):
    """Conservative cell-average transfer; preserves total inverted population."""
    old = np.linspace(0, 1, pop.shape[-1] + 1)
    new = np.linspace(0, 1, cells + 1)
    overlap = np.maximum(
        0,
        np.minimum(new[1:, None], old[None, 1:])
        - np.maximum(new[:-1, None], old[None, :-1]),
    )
    mapped = pop @ overlap.T * cells
    assert np.allclose(mapped.mean(axis=-1), pop.mean(axis=-1), rtol=1e-12, atol=1e-14)
    return mapped


def pump_initial(c, pump, oc, n=2048, dt=0.5, energy_pJ=0.01, seed=0, noisy=False):
    """Pump-only steady EDF inversion plus explicit weak optical seed (not ASE)."""
    a, pop, q = map_initial(c, [oc], n, dt)
    hs = 6.62607015e-34 * 299792458 / (c["lambda_p_nm"] * 1e-9)
    ions = c["ion_density_m3"] * c["doped_area_m2"]
    alpha = c["alpha_p_m"]

    def equilibrium(p):
        rate = alpha * p / (hs * ions)
        return rate / (1 / c["upper_lifetime_s"] + rate)

    length = c["segments"][1, 0]
    cells = pop.shape[-1]
    dz = length / cells
    solution = solve_ivp(
        lambda z, p: -alpha * (1 - equilibrium(p)) * p,
        [0, length],
        [pump * 10 ** (-c["pump_path_dB"] / 10)],
        dense_output=True,
        rtol=1e-10,
        atol=1e-14,
    )
    if not solution.success:
        raise NumericalLimit("pump-only initialization failed")
    # Gauss quadrature returns cell averages, compatible with conservative remapping.
    x, w = np.polynomial.legendre.leggauss(8)
    z = (np.arange(cells)[:, None] + 0.5 + 0.5 * x) * dz
    pop[0] = (equilibrium(solution.sol(z.ravel())[0]).reshape(cells, 8) @ w) / 2
    if noisy:
        rng = np.random.default_rng(seed)
        a = rng.normal(size=a.shape) + 1j * rng.normal(size=a.shape)
        f = np.fft.fftfreq(n, dt)
        a = ifft(fft(a, axis=-1) * np.exp(-((f / 0.02) ** 2)), axis=-1)
    a *= np.sqrt(energy_pJ / (np.sum(abs(a) ** 2) * dt))
    return a, pop, q


class AdaptiveEngine(BatchEngine):
    """Strang step-doubling for passive segments; EDF mesh is independently set."""

    def __init__(self, c, dt, n, pump, oc, options):
        super().__init__(c, dt, n, np.array([pump]), np.array([oc]))
        self.options = options
        self.worst_edges = np.zeros(2)
        self.accepted_steps = 0
        self.rejected_steps = 0
        self.max_local_error = 0.0

    def inspect(self, a):
        edges = health(a, self.dt)
        self.worst_edges = np.maximum(self.worst_edges, edges)
        if (
            edges[0] > self.options["time_warning"]
            or edges[1] > self.options["spectral_warning"]
        ):
            raise BoundaryRefinement(self.worst_edges)

    def fiber(self, a, index, pump, pop):
        if index == 1:
            original_population = pop.copy()
            result = super().fiber(a, index, pump.copy(), pop)
            self.inspect(result[0])
            fine_c = deepcopy(self.c)
            fine_c["max_step_m"] /= 2
            fine = BatchEngine(fine_c, self.dt, self.n, self.pumps, self.oc)
            fine_pop = remap_population(original_population, fine.e.ops[1][0])
            fine_result = fine.fiber(a.copy(), index, pump.copy(), fine_pop)
            field_error = float(
                np.linalg.norm(result[0] - fine_result[0])
                / max(np.linalg.norm(fine_result[0]), 1e-250)
            )
            population_error = float(
                np.max(
                    abs(
                        remap_population(result[2], fine_result[2].shape[-1])
                        - fine_result[2]
                    )
                )
            )
            self.edf_errors = (field_error, population_error)
            if (
                field_error > self.options["edf_rtol"]
                or population_error > self.options["edf_population_atol"]
            ):
                raise EDFRefinement(
                    f"EDF field error {field_error:g}; population error {population_error:g}"
                )
            self.worst_edges = np.maximum(self.worst_edges, health(result[0], self.dt))
            return result
        length, b2, b3, gamma, beat, angle, dgd = self.c["segments"][index]
        w = self.e.w
        D = 1j * b2 * w * w / 2 - 1j * b3 * w**3 / 6
        b = 1j * (beat + dgd * w) / 2
        linear = np.array([D + b, D - b])
        R = self.e.ops[index][3]
        a = R.T @ a

        def strang(field, h):
            half = np.exp(linear * h / 2)
            field = ifft(half * fft(field, axis=-1), axis=-1)
            field = kerr(field, gamma * h)
            return ifft(half * fft(field, axis=-1), axis=-1)

        z = 0.0
        h = min(length, self.c["passive_step_m"])
        while z < length - 1e-12:
            peak = float(np.sum(abs(a) ** 2, axis=1).max())
            phase_step = self.options["nonlinear_phase_max"] / max(
                abs(gamma) * peak, 1e-100
            )
            h = min(h, length - z, phase_step)
            if h < self.options["z_min_m"]:
                raise NumericalLimit("passive z step below resource floor")
            full = strang(a, h)
            half = strang(strang(a, h / 2), h / 2)
            error = float(
                np.linalg.norm(half - full) / max(3 * np.linalg.norm(half), 1e-250)
            )
            tolerance = self.options["z_rtol"] * h / length
            if not np.isfinite(error):
                raise NumericalLimit("nonfinite SSFM local error")
            if error <= tolerance:
                a = half
                z += h
                self.accepted_steps += 1
                self.max_local_error = max(self.max_local_error, error)
                self.inspect(a)
            else:
                self.rejected_steps += 1
            h *= np.clip(0.9 * (tolerance / max(error, 1e-30)) ** (1 / 3), 0.2, 2.0)
            h = min(h, self.c["passive_step_m"])
        return R @ a, pump, pop


class AdaptiveSolver:
    def __init__(self, c, pump, oc, dt=0.5, n=2048, options=None):
        self.c = deepcopy(c)
        self.pump = pump
        self.oc = oc
        self.dt = dt
        self.n = n
        self.options = {**DEFAULTS, **(options or {})}
        rep = 299792458 / (self.c["group_index"] * 20.42)
        if (
            dt <= 0
            or n < 16
            or n % 2
            or pump < 0
            or not 0 < oc < 1
            or n > self.options["max_samples"]
            or n * dt > self.options["max_window_ps"]
            or n * dt > 1e12 / rep - 40 * self.c["sa_recovery_ps"]
        ):
            raise ValueError("Invalid cavity grid, pump, coupling or resource limits")
        if any(
            self.options[k] <= 0
            for k in [
                "z_rtol",
                "z_min_m",
                "time_warning",
                "spectral_warning",
                "edf_rtol",
                "edf_population_atol",
            ]
        ):
            raise ValueError("Numerical tolerances must be positive")
        self.events = []
        self.round = 0
        self.time_shift_ps = 0.0
        self.grid_generation = 0
        self._build()

    def _build(self):
        self.engine = AdaptiveEngine(
            self.c, self.dt, self.n, self.pump, self.oc, self.options
        )

    def step(self, a, pop, q, frozen=False):
        """Only a boundary-clean round is accepted; rejected trials cannot age gain."""
        self.c["gain_mode"] = "frozen" if frozen else "dynamic"
        self.engine.c["gain_mode"] = self.c["gain_mode"]
        clean, shift = recenter(a, self.dt)
        for retry in range(self.options["max_retries"] + 1):
            self.checkpoint = (clean.copy(), pop.copy(), q.copy())
            self.engine.worst_edges = np.zeros(2)
            self.engine.accepted_steps = 0
            self.engine.rejected_steps = 0
            try:
                result = self.engine.step(clean.copy(), pop.copy(), q.copy())
                edges = np.maximum(
                    self.engine.worst_edges,
                    np.maximum(health(result[0], self.dt), health(result[1], self.dt)),
                )
                if not all(np.isfinite(x).all() for x in result):
                    raise NumericalLimit("nonfinite state")
            except BoundaryRefinement as exc:
                result = None
                edges = exc.edges
            except EDFRefinement as exc:
                new_step = self.c["max_step_m"] / 2
                if (
                    new_step < self.options["min_edf_step_m"]
                    or retry == self.options["max_retries"]
                ):
                    raise NumericalLimit(
                        "EDF refinement resource floor: " + str(exc)
                    ) from exc
                self.events.append(
                    dict(
                        round=self.round + 1,
                        action="rollback_edf_refine",
                        old_step_m=self.c["max_step_m"],
                        new_step_m=new_step,
                        reason=str(exc),
                    )
                )
                self.c["max_step_m"] = new_step
                self._build()
                pop = remap_population(pop, self.engine.e.ops[1][0])
                self.grid_generation += 1
                continue
            time_bad = edges[0] > self.options["time_warning"]
            spectral_bad = edges[1] > self.options["spectral_warning"]
            if not time_bad and not spectral_bad:
                self.round += 1
                self.time_shift_ps += shift
                self.last_edges = edges
                return result
            finer = bool(spectral_bad)
            wider = bool(time_bad)
            new_dt = self.dt / (2 if finer else 1)
            new_n = self.n * (2 if finer else 1) * (2 if wider else 1)
            window = new_dt * new_n
            if (
                retry == self.options["max_retries"]
                or new_dt < self.options["min_dt_ps"]
                or new_n > self.options["max_samples"]
                or window > self.options["max_window_ps"]
                or window > 1e12 / self.engine.e.rep - 40 * self.c["sa_recovery_ps"]
            ):
                raise NumericalLimit(
                    "adaptive boundary resource limit; checkpoint remains unaccepted"
                )
            self.events.append(
                dict(
                    round=self.round + 1,
                    action="rollback_refine",
                    time_edge=float(edges[0]),
                    spectral_edge=float(edges[1]),
                    old_dt=self.dt,
                    new_dt=new_dt,
                    old_n=self.n,
                    new_n=new_n,
                )
            )
            clean, self.dt = regrid(clean, self.dt, finer, wider)
            self.n = clean.shape[-1]
            self.grid_generation += 1
            self._build()
        raise NumericalLimit("unreachable retry limit")
