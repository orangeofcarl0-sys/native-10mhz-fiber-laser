"""Physical parameters in m, ps, W and pJ; generated data go to outputs/."""

import json
import os
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("LASER_OUTPUT_DIR", str(PROJECT / "outputs"))).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
OC = np.arange(2, 10) / 10
PUMPS = np.array([5, 10, 15, 20, 30, 50, 100, 150]) / 1000
GDD = np.array([-0.1, 0, 0.1, 0.2, 0.3])
TOPOLOGIES = ["OC_CNT", "CNT_OC"]


def config(topology, gdd):
    if topology not in TOPOLOGIES:
        raise ValueError(f"Unknown topology: {topology}")
    c = json.loads(
        (PROJECT / "parameters/effective_parameters.json").read_text(encoding="utf8")
    )
    c["segments"] = np.array(c.pop("segments_by_GDD")["0.2"], dtype=float)
    c.pop("segment_columns")
    c["pc_turns"] = np.array(c["pc_turns"])
    c["pc_angles_rad"] = np.array(c["pc_angles_rad"])
    c.update(
        topology=topology,
        gain_mode="dynamic",
        max_step_m=0.1,
        passive_step_m=0.5,
        map_EDF_m=1.5,
        map_GDD_ps2=float(gdd),
    )
    le = 1.5
    ln = (gdd - 0.0459 * le + 0.0217 * (20.42 - le)) / 0.1217
    ls = 20.42 - le - ln
    c["segments"][4, 0] = ls - 0.87
    c["segments"][5, 0] = ln
    if np.any(c["segments"][:, 0] <= 0):
        raise ValueError("Requested GDD gives a nonpositive segment length")
    assert abs(c["segments"][:, 0].sum() - 20.42) < 1e-10
    assert abs(np.dot(c["segments"][:, 0], c["segments"][:, 1]) - gdd) < 1e-10
    return c


def map_initial(c, oc, n=2048, dt=0.5):
    """Identical paired-topology initial guesses, not equilibrium laser solutions.

    Returns field[case, polarization, time] in sqrt(W), EDF inversion[case, cell],
    and CNT absorption[case]. No energy normalization occurs after initialization.
    """
    oc = np.atleast_1d(np.asarray(oc, dtype=float))
    if np.any((oc <= 0) | (oc >= 1)):
        raise ValueError("Output coupling must be strictly between zero and one")
    cnt = 1 - c["sa_modulation"] - c["sa_nonsaturable"]
    post = cnt * 10 ** (
        -(c["splice_connector_dB"] + c["hybrid_IL_dB"] + 0.5 * c["hybrid_PDL_dB"]) / 10
    )
    passive = (1 - oc) * post * 10 ** (-c["oc_excess_dB"] / 10)
    inversion = (-np.log(passive) / 1.5 + c["alpha_s_m"]) / (
        c["alpha_s_m"] + c["emission_s_m"]
    )
    assert np.all((inversion > 0) & (inversion < 1))
    t = (np.arange(n) - n / 2) * dt
    shape = np.exp(-t * t / (4 * 100))
    energy = 300 * (1 - oc) / oc * post
    a = np.zeros((len(oc), 2, n), complex)
    a[:, 0] = np.sqrt(energy / (np.sum(shape**2) * dt))[:, None] * shape
    cells = int(np.ceil(c["segments"][1, 0] / c["max_step_m"]))
    pop = np.repeat(inversion[:, None], cells, axis=1)
    return a, pop, np.full(len(oc), c["sa_modulation"])
