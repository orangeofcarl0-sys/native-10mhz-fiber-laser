"""Compare matched continuations in 1.024 and 2.048 ns windows."""

from pathlib import Path
import base64
import json
import numpy as np
from scipy.io import loadmat
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ROOT

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 12,
    }
)


def figure(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=150)
    plt.savefig(FIG / f"{name}.svg")
    plt.close()
    return (
        '<img alt="'
        + name
        + '" src="data:image/png;base64,'
        + base64.b64encode((FIG / f"{name}.png").read_bytes()).decode()
        + '">'
    )


def main():
    suffixes = ["", "_wide", "_refined"]
    maps = [
        loadmat(ROOT / f"long_g03_c11{suffix}.mat", simplify_cells=True)
        for suffix in suffixes
    ]
    summaries = [
        json.loads((ROOT / f"long_g03_c11{suffix}.json").read_text())
        for suffix in suffixes
    ]
    html = [
        "<p>本次固定 OC→CNT、OC 30%、净 GDD +0.2 ps²、pump 20 mW，从细网格第600圈腔内场继续。原始能量及±10%扰动分别计算；两组时间窗为1.024 ns与2.048 ns，Δt均为0.125 ps，无源传播步长均为0.25 m。加宽仅两端补零，不改变初始能量、反转或CNT状态。</p>"
    ]
    rows = []
    for width, suffix, m, summary in zip([1024, 2048, 1024], suffixes, maps, summaries):
        for i, r in enumerate(summary):
            tr = m["trace"][: int(m["completed"][i]), i]
            tail = tr[-500:]
            valid = (tr[:, 3] <= 1e-6) & (tr[:, 4] <= 1e-8)
            exits = np.flatnonzero(valid & ((tr[:, 0] < 100) | (tr[:, 0] > 500)))
            multi = np.flatnonzero(valid & (tr[:, 1] > 1))
            rows.append(
                dict(
                    window_ps=width,
                    dt_ps=float(m["dt"]),
                    variant=suffix or "baseline",
                    first_out_round=int(exits[0]) + 601 if len(exits) else None,
                    first_multi_round=int(multi[0]) + 601 if len(multi) else None,
                    **r,
                    tail_peaks_min=int(tail[:, 1].min()),
                    tail_peaks_max=int(tail[:, 1].max()),
                    time_edge=float(tr[-1, 3]),
                    spectral_edge=float(tr[-1, 4]),
                )
            )
        time = (600 + np.arange(1, len(m["trace"]) + 1)) / 10
        for column, label, short in [
            (0, "输出单圈能量 / nJ", "energy"),
            (1, "全窗显著峰数", "peaks"),
            (5, "EDF平均反转比例", "inversion"),
        ]:
            plt.figure(figsize=(12, 5.4))
            for i, factor in enumerate([0.9, 1.0, 1.1]):
                k = int(m["completed"][i])
                tr = m["trace"][:k, i]
                y = tr[:, column] / 1000 if column == 0 else tr[:, column]
                plt.plot(time[:k], y, label=f"初始能量 ×{factor:g}", lw=1.4)
            if column == 0:
                plt.axhspan(
                    0.1, 0.5, color="#208c75", alpha=0.12, label="目标能量 0.1–0.5 nJ"
                )
                plt.yscale("log")
            plt.xlabel("从原始初始化起算的演化时间 / μs（约10圈/μs）")
            plt.ylabel(label)
            plt.title(
                f'30% OC、+0.2 ps²、20 mW：{width/1000:g} ns时间窗，Δt={float(m["dt"]):g} ps'
            )
            plt.grid(alpha=0.2)
            plt.legend()
            html.append(figure(f"best_candidate_{width}{suffix}_{short}"))
    m = maps[2]
    field = m["out"][1]
    power = np.sum(abs(field) ** 2, axis=0)
    t = m["t"]
    center = t[np.argmax(power)]
    plt.figure(figsize=(12, 5.4))
    plt.plot(t - center, power, lw=1)
    plt.xlabel("相对主峰时间 / ps")
    plt.ylabel("瞬时输出功率 / W")
    plt.title("联合细化：总第2100圈未扰动分支的全窗输出场")
    plt.grid(alpha=0.2)
    html.append(figure("best_candidate_refined_output"))
    spectrum = np.fft.fftshift(np.sum(abs(np.fft.fft(field, axis=-1)) ** 2, axis=0))
    freq = np.fft.fftshift(np.fft.fftfreq(len(t), float(m["dt"])))
    plt.figure(figsize=(12, 5.4))
    plt.plot(freq, 10 * np.log10(np.maximum(spectrum / spectrum.max(), 1e-12)), lw=1)
    plt.xlabel("相对光学载频偏移 / THz")
    plt.ylabel("输出场光谱包络 / dB")
    plt.ylim(-100, 3)
    plt.title("联合细化：与时域快照对应的输出场光谱包络")
    plt.grid(alpha=0.2)
    html.append(figure("best_candidate_refined_spectrum"))
    html.append(
        "<p>以上为同一未扰动分支的末圈诊断快照，不能作为稳定光源规格。光谱是1.024 ns局部时间窗的包络，不解析10 MHz光梳线。多峰数仍依赖所用分辨率与10%显著性阈值；本次用其判断尚未保持单脉冲，不将精确峰数当作已收敛的物理预测。</p>"
    )
    comparisons = []
    for i in range(3):
        k = min(int(m["completed"][i]) for m in maps[:2])
        x = maps[0]["trace"][:k, i, 0]
        y = maps[1]["trace"][:k, i, 0]
        # Compare only pre-stop rows, not the boundary-triggering sample.
        k = max(1, k - 20)
        x = x[:k]
        y = y[:k]
        comparisons.append(
            dict(
                factor=[0.9, 1.0, 1.1][i],
                compared_rounds=k,
                max_relative_energy_difference=float(
                    np.max(abs(x - y) / np.maximum(abs(y), 1e-100))
                ),
                first_200_relative=float(
                    np.max(abs(x[:200] - y[:200]) / np.maximum(abs(y[:200]), 1e-100))
                ),
            )
        )
    refinement = []
    for i in range(3):
        x = maps[0]["trace"][:200, i, 0]
        y = maps[2]["trace"][:200, i, 0]
        refinement.append(
            dict(
                factor=[0.9, 1.0, 1.1][i],
                first_200_relative=float(
                    np.max(abs(x - y) / np.maximum(abs(y), 1e-100))
                ),
            )
        )
    html.append(
        "<p>第三组保持1.024 ns窗口，将时间步长减至0.0625 ps、无源光纤最大步长减至0.125 m（EDF仍为0.1 m），最多新增1500圈。此组是时间/无源空间联合细化，不单独区分两类误差。前200个新增圈能量相对差为："
        + ", ".join(f"{r['first_200_relative']:.3g}" for r in refinement)
        + "。</p>"
    )
    html.append(
        '<div class="scroll"><table><tr>'
        + "".join(
            "<th>" + x + "</th>"
            for x in [
                "时间窗/ps",
                "Δt/ps",
                "能量倍率",
                "新增圈数",
                "停止原因",
                "末500圈能量/nJ",
                "能量CV",
                "末500圈峰数",
            ]
        )
        + "</tr>"
    )
    for r in rows:
        cells = [
            r["window_ps"],
            r["dt_ps"],
            r["factor"],
            r["rounds"],
            r["status"],
            f"{r['energy_min_pJ']/1000:.4g}–{r['energy_max_pJ']/1000:.4g}",
            f"{r['energy_cv']:.1%}",
            f"{r['tail_peaks_min']}–{r['tail_peaks_max']}",
        ]
        html.append(
            "<tr>" + "".join("<td>" + str(x) + "</td>" for x in cells) + "</tr>"
        )
    html.append("</table></div>")
    html.append(
        '<p><b>本候选尚未通过稳定单脉冲验证。</b>下表只在当圈时间/频谱边界均通过时，记录首次能量离开0.1–0.5 nJ及首次出现多个显著峰的位置；圈数从原始初始化起算。</p><div class="scroll"><table><tr><th>设置</th><th>能量倍率</th><th>首次离开目标带/圈</th><th>首次多峰/圈</th></tr>'
    )
    for r in rows:
        html.append(
            f'<tr><td>{r["window_ps"]} ps / Δt {r["dt_ps"]} ps</td><td>{r["factor"]}</td><td>{r["first_out_round"]}</td><td>{r["first_multi_round"]}</td></tr>'
        )
    html.append(
        "</table></div><p>表中CV是沿演化轨迹的标准差/均值，不是实验重复测量误差。数值越界停止后的轨迹未知；停止前的能量离带、波形变化和多峰可以记录，但不能把停止原因直接等同于物理不稳定机制。</p>"
    )
    html.append(
        "<p>两时间窗前200个新增圈的能量最大相对差（按0.9、1.0、1.1倍率）为："
        + ", ".join(f"{r['first_200_relative']:.3g}" for r in comparisons)
        + "。后期差异和边界触发需要结合完整轨迹判读，不能将局部数值吻合当作长期稳态认证。</p>"
    )
    baseline = rows[1]
    fine = rows[7]
    html.insert(
        1,
        f'<p class="callout"><b>本次未新增可确认稳定的单脉冲工作点。</b>未扰动状态在两时间窗中均于总第{baseline["first_out_round"]}圈首次离开目标能量带。联合细化网格完成新增{fine["rounds"]}圈，末500圈能量范围{fine["energy_min_pJ"]/1000:.4g}–{fine["energy_max_pJ"]/1000:.4g} nJ，能量CV为{fine["energy_cv"]:.1%}，显著峰数范围{fine["tail_peaks_min"]}–{fine["tail_peaks_max"]}。能量变化出现在数值边界触发之前；但后续长期吸引态仍未确定，不能断言此参数永远无法锁模。</p>',
    )
    (ROOT / "best_candidate_validation.json").write_text(
        json.dumps(
            dict(
                rows=rows,
                window_comparison=comparisons,
                refinement_comparison=refinement,
            ),
            indent=2,
        ),
        encoding="utf8",
    )
    (ROOT / "best_candidate_validation.html").write_text("".join(html), encoding="utf8")
    print(
        json.dumps(
            dict(
                rows=rows,
                window_comparison=comparisons,
                refinement_comparison=refinement,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
