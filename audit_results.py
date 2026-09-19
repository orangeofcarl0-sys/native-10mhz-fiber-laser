"""Check complete Cartesian coverage, recorded observables and paired initial states."""

import json, math
import numpy as np
from scipy.io import loadmat
from config import ROOT, OC, PUMPS, GDD, TOPOLOGIES

result = {"grids": [], "paired_initial_checks": 0}
for prefix, dt in [("", 0.5), ("fine_", 0.125)]:
    allrows = []
    for g in range(10):
        name = f"{prefix}group_{g:02d}"
        rows = json.loads((ROOT / f"{name}.json").read_text(encoding="utf8"))
        m = loadmat(
            ROOT / f"{name}.mat",
            simplify_cells=True,
            variable_names=["c", "dt", "oc", "pumps", "trace", "completed"],
        )
        assert len(rows) == 64 and len({r["id"] for r in rows}) == 64
        assert m["dt"] == dt and abs(m["c"]["segments"][:, 0].sum() - 20.42) < 1e-10
        assert (
            abs(np.dot(m["c"]["segments"][:, 0], m["c"]["segments"][:, 1]) - GDD[g % 5])
            < 1e-10
        )
        for i, r in enumerate(rows):
            assert r["topology"] == TOPOLOGIES[g // 5] and r["GDD_ps2"] == GDD[g % 5]
            assert (
                r["oc"] == OC[i // 8]
                and abs(r["pump_mW"] - PUMPS[i % 8] * 1000) < 1e-10
            )
            assert all(not isinstance(v, float) or math.isfinite(v) for v in r.values())
            tr = m["trace"][: int(m["completed"][i]), i]
            assert np.isclose(tr[-1, 0], r["energy_pJ"], rtol=1e-12, atol=0)
            assert int(tr[-1, 3]) == r["peaks"] and len(tr) == r["rounds"]
            if r["numerical_pass"]:
                assert (
                    len(tr) == 600
                    and np.max(tr[-64:, 5]) < 1e-6
                    and np.max(tr[-64:, 6]) < 1e-8
                )
            assert r["energy_range_pass"] == bool(
                np.min(tr[-100:, 0]) >= 100 and np.max(tr[-100:, 0]) <= 500
            )
            assert r["single_tail"] == bool(np.all(tr[-64:, 3] == 1))
        allrows.extend(rows)
    assert (
        len({(r["topology"], r["oc"], r["GDD_ps2"], r["pump_mW"]) for r in allrows})
        == 640
    )
    for g in range(5):
        a = loadmat(
            ROOT / f"{prefix}group_{g:02d}.mat",
            simplify_cells=True,
            variable_names=["initial_a", "initial_pop"],
        )
        b = loadmat(
            ROOT / f"{prefix}group_{g+5:02d}.mat",
            simplify_cells=True,
            variable_names=["initial_a", "initial_pop"],
        )
        assert np.array_equal(a["initial_a"], b["initial_a"]) and np.array_equal(
            a["initial_pop"], b["initial_pop"]
        )
        result["paired_initial_checks"] += 1
    result["grids"].append(
        dict(
            prefix=prefix,
            points=len(allrows),
            boundary_pass=sum(r["numerical_pass"] for r in allrows),
            target_single=sum(
                r["numerical_pass"] and r["energy_range_pass"] and r["single_tail"]
                for r in allrows
            ),
            screen_candidate=sum(r["screen_candidate"] for r in allrows),
        )
    )
result["all_passed"] = True
(ROOT / "audit_results.json").write_text(json.dumps(result, indent=2), encoding="utf8")
print(result)
