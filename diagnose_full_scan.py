"""Read the completed scan and probe one additional round; no stability claim.

LASER_SCAN_DIR: completed paired-grid run. LASER_OUTPUT_DIR: diagnostic outputs.
Stored round-600 statistics and round-601 probes are explicitly separated.
"""

import os, json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import ROOT, config
from spectral_engine import SpectralEngine


def main():
    source = Path(os.environ["LASER_SCAN_DIR"])
    rows = sum(
        [
            json.loads((source / f"fine_group_{g:02d}.json").read_text())
            for g in range(10)
        ],
        [],
    )
    assert len(rows) == 640
    valid = [r for r in rows if r["numerical_pass"]]
    targets = sorted(
        [r for r in valid if r["energy_range_pass"] and r["single_tail"]],
        key=lambda r: r["energy_cv"],
    )
    selected = targets + [
        next(r for r in rows if r["id"] == name)
        for name in ["g08_c53", "g00_c49", "g01_c50", "g01_c41"]
    ]
    result = {
        "scope": "postprocessing plus one-round CPU FP64 diagnostic; not continuation or mechanism intervention",
        "selected": [],
        "boundary_counts": {},
    }
    for key in ["pump_mW", "oc", "GDD_ps2"]:
        result["boundary_counts"][key] = [
            {
                "value": value,
                "boundary": sum(
                    r["status"] == "spectral_boundary" for r in rows if r[key] == value
                ),
                "total": sum(r[key] == value for r in rows),
            }
            for value in sorted(set(r[key] for r in rows))
        ]
    traces = {}
    for group in sorted(set(r["group"] for r in selected)):
        data = loadmat(
            source / f"fine_group_{group:02d}.mat",
            variable_names=["trace", "a", "pop", "q", "rep"],
        )
        group_rows = [r for r in selected if r["group"] == group]
        ids = [r["index"] for r in group_rows]
        c = config(group_rows[0]["topology"], group_rows[0]["GDD_ps2"])
        engine = SpectralEngine(
            c,
            0.125,
            8192,
            np.array([r["pump_mW"] for r in group_rows]) / 1000,
            np.array([r["oc"] for r in group_rows]),
        )
        original = engine.absorber
        absorption = {}

        def instrument(a, q, c, dt):
            out, q, sa = original(a, q, c, dt)
            pin = np.sum(abs(a) ** 2, axis=1)
            pout = np.sum(abs(out) ** 2, axis=1)
            absorption["transmission"] = pout.sum(axis=1) / pin.sum(axis=1)
            return out, q, sa

        engine.absorber = instrument
        pop = np.ascontiguousarray(data["pop"][ids])
        _, _, next_pop, _, sa = engine.step(
            np.ascontiguousarray(data["a"][ids]),
            pop.copy(),
            np.ascontiguousarray(data["q"].ravel()[ids]),
        )
        assert np.isfinite(sa).all() and np.all(
            (absorption["transmission"] > 0) & (absorption["transmission"] < 1)
        )
        for j, r in enumerate(group_rows):
            tr = data["trace"][: r["rounds"], r["index"]]
            traces[r["id"]] = tr
            ratio = r["CNT_peak_W"] / 40
            item = dict(r)
            item.update(
                saturation_ratio_600=ratio,
                q_min_600=float(tr[-1, 9]),
                bleach_fraction_600=float((0.05 - tr[-1, 9]) / 0.05),
                quasi_q_min_600=float(0.05 / (1 + ratio)),
                energy_cv_blocks100=[
                    float(np.std(v[:, 0]) / np.mean(v[:, 0]))
                    for v in np.array_split(tr, 6)
                ],
                tail_energy_endpoint_change_relative=float(
                    (tr[-1, 0] - tr[-100, 0]) / np.mean(tr[-100:, 0])
                ),
                probe601_tau_max_us=float(engine.last_tau[j] * 1e6),
                probe601_tau_max_rounds=float(engine.last_tau[j] * engine.e.rep),
                probe601_inversion_gap_max=float(engine.last_gap[j]),
                probe601_population_step_max=float(np.max(abs(next_pop[j] - pop[j]))),
                probe601_cnt_energy_transmission=float(absorption["transmission"][j]),
                probe601_cnt_effective_bleach_pp=float(
                    100 * (absorption["transmission"][j] - 0.85)
                ),
            )
            result["selected"].append(item)
    (ROOT / "mechanism_diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
    )
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 12,
        }
    )

    def save(name):
        plt.tight_layout()
        plt.savefig(ROOT / name, dpi=160)
        plt.close()

    names = ["g03_c11", "g08_c52", "g08_c53"]
    for column, ylabel, filename in [
        (0, "输出能量 / nJ", "energy_history.png"),
        (10, "EDF 最大瞬时反转平衡偏差", "gain_gap_history.png"),
        (9, "CNT 可饱和损耗的漂白比例 / %", "cnt_history.png"),
    ]:
        plt.figure(figsize=(10, 5))
        for name in names:
            y = traces[name][:, column].copy()
            if column == 0:
                y /= 1000
            if column == 9:
                y = (0.05 - y) / 0.05 * 100
            plt.plot(np.arange(1, len(y) + 1), y, label=name)
        if column == 0:
            plt.axhspan(0.1, 0.5, color="teal", alpha=0.1, label="目标能量范围")
        plt.xlabel("传播圈数（600圈约60 μs）")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(alpha=0.2)
        save(filename)
    plt.figure(figsize=(10, 5))
    for label, subset, marker in [
        ("末段单峰、目标范围", targets, "o"),
        (
            "末段单峰、目标范围外",
            [r for r in valid if r["single_tail"] and not r["energy_range_pass"]],
            "^",
        ),
        ("末段非单峰", [r for r in valid if not r["single_tail"]], "x"),
    ]:
        plt.scatter(
            [r["CNT_peak_W"] / 40 for r in subset],
            [r["energy_cv"] * 100 for r in subset],
            label=label,
            marker=marker,
            alpha=0.75,
        )
    plt.axvline(1, color="gray", ls="--", label="P峰/Psat=1（不是锁模门槛）")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("第600圈 CNT 峰值功率 / 40 W")
    plt.ylabel("末100圈能量 CV / %")
    plt.legend()
    plt.grid(alpha=0.2)
    save("saturation_vs_cv.png")
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in [
                        "id",
                        "probe601_tau_max_us",
                        "probe601_inversion_gap_max",
                        "probe601_cnt_effective_bleach_pp",
                    ]
                }
                for r in result["selected"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
