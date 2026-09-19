"""Audit all new trajectories and compare archived maps without mixing data sources.

LASER_PREVIOUS_DIR points to the prior raw map directory. Only compact summaries
are published; raw MAT trajectories stay in the isolated LASER_OUTPUT_DIR.
"""

import csv, hashlib, json, os
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from config import ROOT


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def counts(rows):
    valid = [r for r in rows if r["numerical_pass"]]
    single = [r for r in valid if r["energy_range_pass"] and r["single_tail"]]
    return dict(
        cases=len(rows),
        completed_600=sum(r["rounds"] == 600 for r in rows),
        numerical_pass=len(valid),
        target_single=len(single),
        short_screen=sum(r["screen_candidate"] for r in rows),
        status_counts=dict(Counter(r["status"] for r in rows)),
        actual_case_rounds=sum(r["rounds"] for r in rows),
    )


def category(row):
    if not row["numerical_pass"]:
        return 0
    if not row["energy_range_pass"]:
        return 1
    if not row["single_tail"]:
        return 2
    return 4 if row["screen_candidate"] else 3


def main():
    previous = Path(os.environ["LASER_PREVIOUS_DIR"])
    manifest = read(ROOT / "run_manifest.json")
    assert manifest["status"] == "complete" and len(manifest["completed_groups"]) == 20
    result = dict(
        scope="640 physical combinations, two grids, 600RT ceiling; no stable-solution certification",
        grids={},
        files={},
        all_inputs_match=True,
    )
    combined = []
    changes = []
    for tag, label in [("", "coarse"), ("fine_", "fine")]:
        current = []
        prior = []
        early = []
        late = []
        first = []
        first5 = []
        for group in range(10):
            name = f"{tag}group_{group:02d}"
            rows = read(ROOT / f"{name}.json")
            old = read(previous / f"{name}.json")
            assert len(rows) == len(old) == 64
            a = loadmat(
                ROOT / f"{name}.mat",
                variable_names=["trace", "initial_a", "initial_pop", "completed"],
                simplify_cells=True,
            )
            b = loadmat(
                previous / f"{name}.mat",
                variable_names=["trace", "initial_a", "initial_pop", "completed"],
                simplify_cells=True,
            )
            same = np.array_equal(a["initial_a"], b["initial_a"]) and np.array_equal(
                a["initial_pop"], b["initial_pop"]
            )
            result["all_inputs_match"] &= same
            assert same, f"Initialization mismatch: {name}"
            for x, y in zip(rows, old):
                assert x["id"] == y["id"]
                for key in ["topology", "oc", "pump_mW", "GDD_ps2"]:
                    assert x[key] == y[key], (name, key)
                i = x["index"]
                steps = min(x["rounds"], y["rounds"])
                assert int(a["completed"][i]) == x["rounds"]
                new_energy = a["trace"][:steps, i, 0]
                old_energy = b["trace"][:steps, i, 0]
                relative = abs(new_energy - old_energy) / np.maximum(
                    abs(old_energy), 1e-100
                )
                first.append(float(relative[0]))
                first5.append(float(np.max(relative[: min(5, steps)])))
                early.append(float(np.max(relative[: min(20, steps)])))
                late.append(float(np.max(relative)))
                if (
                    x["status"] != y["status"]
                    or x["numerical_pass"] != y["numerical_pass"]
                    or category(x) != category(y)
                ):
                    changes.append(
                        dict(
                            grid=label,
                            id=x["id"],
                            old_status=y["status"],
                            new_status=x["status"],
                            old_rounds=y["rounds"],
                            new_rounds=x["rounds"],
                            old_single=y["single_tail"],
                            new_single=x["single_tail"],
                            old_numerical=y["numerical_pass"],
                            new_numerical=x["numerical_pass"],
                            first20_energy_max_relative=early[-1],
                            common_history_energy_max_relative=late[-1],
                            old_mean_energy_nJ=y["energy_mean_pJ"] / 1000,
                            new_mean_energy_nJ=x["energy_mean_pJ"] / 1000,
                            old_energy_cv=y["energy_cv"],
                            new_energy_cv=x["energy_cv"],
                            first_difference_above_1e6=(
                                int(np.flatnonzero(relative > 1e-6)[0] + 1)
                                if np.any(relative > 1e-6)
                                else None
                            ),
                            first_difference_above_1percent=(
                                int(np.flatnonzero(relative > 0.01)[0] + 1)
                                if np.any(relative > 0.01)
                                else None
                            ),
                            energy_relative_difference=relative.tolist(),
                        )
                    )
            current.extend(rows)
            prior.extend(old)
            for extension in ["json", "mat"]:
                path = ROOT / f"{name}.{extension}"
                h = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(chunk)
                result["files"][path.name] = dict(
                    bytes=path.stat().st_size, sha256=h.hexdigest()
                )
        assert len({r["id"] for r in current}) == 640
        candidates = sorted(
            [
                r
                for r in current
                if r["numerical_pass"] and r["energy_range_pass"] and r["single_tail"]
            ],
            key=lambda r: r["energy_cv"],
        )
        result["grids"][label] = dict(
            current=counts(current),
            previous=counts(prior),
            same_stop_status=sum(
                x["status"] == y["status"] for x, y in zip(current, prior)
            ),
            same_map_category=sum(
                category(x) == category(y) for x, y in zip(current, prior)
            ),
            same_numerical_flag=sum(
                x["numerical_pass"] == y["numerical_pass"]
                for x, y in zip(current, prior)
            ),
            first_round_energy_max_relative=max(first),
            first5_energy_max_relative=max(first5),
            first20_energy_max_relative=max(early),
            common_history_energy_relative_median=float(np.median(late)),
            common_history_energy_relative_max=max(late),
            target_single_points=candidates,
        )
        combined.extend([dict(grid=label, **r) for r in current])
    result["changed_cases"] = changes
    (ROOT / "full_run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
    )
    (ROOT / "all_1280_results.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf8"
    )
    with (ROOT / "all_1280_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    print(
        json.dumps(
            {k: v["current"] for k, v in result["grids"].items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
