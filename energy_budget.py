"""Independent monochromatic steady gain budget, not a mode-locking solver."""

import json
import numpy as np
from config import config, ROOT, OC, PUMPS

c = config("OC_CNT", 0.2)
oc = np.repeat(OC, 8)
pump = np.tile(PUMPS, 8)
hp = 6.62607015e-34 * 299792458 / (c["lambda_p_nm"] * 1e-9)
hs = 6.62607015e-34 * 299792458 / (c["lambda_s_nm"] * 1e-9)
ions = c["ion_density_m3"] * c["doped_area_m2"]
rep = 299792458 / (c["group_index"] * 20.42)


def propagate(signal, steps):
    """Steady inversion at local powers; RK4 along the 1.5 m EDF."""
    pp = pump * 10 ** (-c["pump_path_dB"] / 10)
    ps = signal.copy()
    dz = 1.5 / steps

    def rhs(p, s):
        n = (c["alpha_p_m"] * p / hp + c["alpha_s_m"] * s / hs) / (
            ions / c["upper_lifetime_s"]
            + c["alpha_p_m"] * p / hp
            + (c["alpha_s_m"] + c["emission_s_m"]) * s / hs
        )
        return (
            -c["alpha_p_m"] * (1 - n) * p,
            (c["emission_s_m"] * n - c["alpha_s_m"] * (1 - n)) * s,
        )

    for j in range(steps):
        ap, as_ = rhs(pp, ps)
        bp, bs = rhs(pp + dz * ap / 2, ps + dz * as_ / 2)
        cp, cs = rhs(pp + dz * bp / 2, ps + dz * bs / 2)
        dp, ds = rhs(pp + dz * cp, ps + dz * cs)
        pp += dz * (ap + 2 * bp + 2 * cp + dp) / 6
        ps += dz * (as_ + 2 * bs + 2 * cs + ds) / 6
    return ps, pp


def solve(transmission, pdl, steps=150):
    passive = (
        (1 - oc)
        * transmission
        * 10
        ** (
            -(c["oc_excess_dB"] + c["splice_connector_dB"] + c["hybrid_IL_dB"] + pdl)
            / 10
        )
    )
    lower = np.full(64, -14.0)
    upper = np.zeros(64)
    gain0 = propagate(10**lower, steps)[0] / 10**lower * passive
    for j in range(45):
        mid = (lower + upper) / 2
        power = 10**mid
        ratio = propagate(power, steps)[0] * passive / power
        lower = np.where(ratio > 1, mid, lower)
        upper = np.where(ratio <= 1, mid, upper)
    signal = 10 ** ((lower + upper) / 2)
    end, remaining = propagate(signal, steps)
    signal = np.where(gain0 > 1, signal, 0)
    end = np.where(gain0 > 1, end, 0)
    output = end * 10 ** (-c["oc_excess_dB"] / 10) * oc
    return output, signal, end, remaining, gain0


if __name__ == "__main__":
    rows = []
    validation = []
    for label, t, pdl in [("loss_high", 0.85, 0.15), ("loss_low", 0.90, 0)]:
        out, start, end, remaining, gain0 = solve(t, pdl)
        fine = solve(t, pdl, 300)[0]
        rel = float(np.max(abs(fine - out) / np.maximum(fine, 1e-20)))
        assert rel < 1e-5, rel
        incident = pump * 10 ** (-c["pump_path_dB"] / 10)
        # Pump photon conversion bound: Psignal_added <= Ppump_absorbed * lambda_p/lambda_s.
        ratio = (end - start) / np.maximum(
            (incident - remaining) * (c["lambda_p_nm"] / c["lambda_s_nm"]), 1e-30
        )
        assert np.max(ratio) < 1 + 1e-6
        validation.append(
            dict(
                bound=label,
                grid_relative_error=rel,
                max_photon_conversion_fraction=float(ratio.max()),
            )
        )
        for i in range(64):
            for topology in ["OC_CNT", "CNT_OC"]:
                power = out[i] * (t if topology == "CNT_OC" else 1)
                rows.append(
                    dict(
                        topology=topology,
                        oc=float(oc[i]),
                        pump_mW=float(pump[i] * 1000),
                        bound=label,
                        CNT_transmission=t,
                        hybrid_PDL_dB=pdl,
                        power_mW=float(power * 1000),
                        energy_nJ=float(power / rep * 1e9),
                        small_signal_roundtrip_gain=float(gain0[i]),
                    )
                )
    (ROOT / "steady_energy_budget.json").write_text(
        json.dumps(dict(rows=rows, validation=validation), indent=2), encoding="utf8"
    )
    print(validation)
