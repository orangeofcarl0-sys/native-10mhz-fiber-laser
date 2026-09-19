"""Short integration checks of state-carrying sweeps, not attractor searches."""

import json
from attractor_search import continuation, multi_seed
from config import ROOT

rows = continuation(
    "OC_CNT",
    0.2,
    0.3,
    [1.0, 2.0],
    search_blocks=0,
    direct_rounds=4,
    n=512,
    name="path_smoke_pump",
    allow_unconverged=True,
)
assert [r["direction"] for r in rows] == ["up", "up", "down", "down"]
assert (
    not rows[0]["seed_from_previous_nonconverged"]
    and rows[1]["seed_from_previous_nonconverged"]
)
assert (
    not rows[2]["seed_from_previous_nonconverged"]
    and rows[3]["seed_from_previous_nonconverged"]
)
gdd = continuation(
    "CNT_OC",
    0.2,
    0.3,
    [0.1, 0.2],
    axis="gdd",
    pump_mW=1.0,
    search_blocks=0,
    direct_rounds=4,
    n=512,
    name="path_smoke_gdd",
    allow_unconverged=True,
)
assert [r["gdd_ps2"] for r in gdd] == [0.1, 0.2, 0.2, 0.1]
seeds = multi_seed(
    pump=0.001, search_blocks=0, direct_rounds=2, n=512, name="path_smoke_seeds"
)
assert all(r["status"] == "insufficient_history" for r in rows + gdd + seeds)
result = dict(
    pump_path_points=len(rows),
    gdd_path_points=len(gdd),
    seed_cases=len(seeds),
    independent_sweep_endpoints=True,
    unconverged_warm_starts_labelled=True,
    passed=True,
)
(ROOT / "search_path_validation.json").write_text(
    json.dumps(result, indent=2), encoding="utf8"
)
print(result)
