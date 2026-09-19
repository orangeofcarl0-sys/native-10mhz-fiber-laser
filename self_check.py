"""Physical invariants and archived-result checks; no MATLAB or GPU required."""

import json
import numpy as np
from config import PROJECT, GDD, OC, config, map_initial
from batch_engine import BatchEngine, kerr
from diagnostics import align

for gdd in GDD:
    c = config("OC_CNT", gdd)
    other = config("CNT_OC", gdd)
    for left, right in zip(map_initial(c, OC), map_initial(other, OC)):
        assert np.array_equal(left, right)
    assert abs(c["segments"][:, 0].sum() - 20.42) < 1e-10
    assert abs(c["segments"][:, 0] @ c["segments"][:, 1] - gdd) < 1e-10

c = config("OC_CNT", 0.2)
a, pop, q = map_initial(c, [0.2, 0.9])
energy = lambda field: np.sum(abs(field) ** 2, axis=(1, 2))
assert np.allclose(energy(kerr(a.copy(), 0.2)), energy(a), rtol=1e-13)
engine = BatchEngine(c, 0.5, 2048, np.array([0.01, 0.03]), np.array([0.2, 0.9]))
passive, _, _ = engine.fiber(a.copy(), 4, engine.pumps.copy(), pop.copy())
assert np.allclose(energy(passive), energy(a), rtol=1e-12)
retained, output = engine.split(a)
assert np.allclose(
    energy(retained) + energy(output),
    energy(a) * 10 ** (-c["oc_excess_dB"] / 10),
    rtol=1e-13,
)
ir, fr, shift = align(a[0], np.roll(a[0], 7, axis=-1) * np.exp(0.8j))
assert ir < 1e-12 and fr < 1e-12 and abs(shift + 7) < 1e-9

rows = json.loads(
    (PROJECT / "results/reference/map_results.json").read_text(encoding="utf8")
)
assert len(rows) == 640
assert len({(r["topology"], r["oc"], r["GDD_ps2"], r["pump_mW"]) for r in rows}) == 640
assert sum(r["numerical_pass"] for r in rows) == 264
assert (
    sum(
        r["numerical_pass"] and r["energy_range_pass"] and r["single_tail"]
        for r in rows
    )
    == 8
)
assert sum(r["screen_candidate"] for r in rows) == 0
print(
    "PASS: paired initialization, cavity constraints, passive/Kerr/OC energy, alignment, 640-case archive"
)
