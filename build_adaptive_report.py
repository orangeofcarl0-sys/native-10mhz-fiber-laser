"""Portable HTML validation appendix; figures are embedded for distribution."""

from pathlib import Path
import base64, json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from config import ROOT, config
from adaptive_solver import pump_initial


def main():
    figures = ROOT / "adaptive_figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 12,
        }
    )

    def read(name):
        return json.loads((ROOT / f"{name}.json").read_text(encoding="utf8"))

    def save(name):
        plt.tight_layout()
        plt.savefig(figures / f"{name}.png", dpi=140)
        plt.savefig(figures / f"{name}.svg")
        plt.close()
        return (
            '<img style="width:100%" alt="'
            + name
            + '" src="data:image/png;base64,'
            + base64.b64encode((figures / f"{name}.png").read_bytes()).decode()
            + '">'
        )

    recovery = read("adaptive_recovery_validation")
    mesh = read("edf_mesh_validation")["rows"]
    pilot = read("adaptive_pilot_checkpoint")
    low = read("adaptive_pilot_lowpump")
    paths = read("search_path_validation")
    html = [
        "<p><b>本次完成P0求解链改造与验证，尚未用新算法重算640点地图，也未新增已认证稳定光源。</b>历史结果保留；下列验证用于判断新求解器是否正确实现所述数值操作。</p>",
        "<p>新流程：泵浦相关初态／多种子与双向延续 → 快场松弛与受限慢增益搜索 → 检查点回退、扩窗／频谱细化 → 无源SSFM步长加倍与EDF双网格控制 → 真实逐圈动力学 → 复场／偏振／纵向反转周期检查。</p>",
        "<p>搜索加速不计入真实时间，也不用于稳定性判断。真实验证需至少累计5个最慢单元的有效增益松弛时间，同时满足局部周期与反转判据。有限局部窗口仍不能证明整个100ns周期内没有其他脉冲。</p>",
        f'<p><b>频谱恢复对照：</b>合成高峰值近频谱边界脉冲触发回退，Δt从0.5减至0.25ps，点数2048增至4096。与直接细网格出发的输出复场相对差为 {recovery["output_field_relative_error"]:.3g}，只接受1个往返，未重复推进增益。这是数值压力测试，不是锁模解。</p>',
    ]
    plt.figure(figsize=(11, 5.3))
    for case, label in [
        ("near_candidate", "目标附近状态"),
        ("boundary_snapshot", "旧边界快照（仅压力测试）"),
        ("multipulse_snapshot", "多峰状态"),
    ]:
        rows = [r for r in mesh if r["case"] == case and r["edf_step_m"] > 0.025]
        plt.plot(
            [r["edf_step_m"] for r in rows],
            [r["field_error_vs_0025"] for r in rows],
            "-o",
            label=label,
        )
    plt.yscale("log")
    plt.xlabel("EDF最大空间步长 / m")
    plt.ylabel("复场相对差（对照0.025 m）")
    plt.title("独立EDF网格检查：保持输入场与时间采样不变")
    plt.grid(alpha=0.2)
    plt.legend()
    html.append(save("edf_grid_errors"))
    c = config("OC_CNT", 0.2)
    c["max_step_m"] = 0.025
    plt.figure(figsize=(11, 5.3))
    for pump in [0.001, 0.005, 0.01, 0.02]:
        _, pop, _ = pump_initial(c, pump, 0.3)
        z = (np.arange(pop.shape[-1]) + 0.5) * 1.5 / pop.shape[-1]
        plt.plot(z, pop[0], label=f"{pump*1000:g} mW")
    plt.xlabel("EDF内位置 / m")
    plt.ylabel("泵独立稳态反转比例")
    plt.title("泵浦相关初始化：未加入信号时的EDF反转分布")
    plt.grid(alpha=0.2)
    plt.legend()
    html.append(save("pump_conditioned_inversion"))
    html.append(
        "<p>泵浦相关初态是EDF预充能后的反转分布，再叠加弱高斯或带限随机种子；不包含ASE，不等价于未泵浦冷腔的真实自启动。</p>"
    )
    tr = pilot["trace"]
    summary = pilot["summary"]
    plt.figure(figsize=(11, 5.3))
    plt.plot(
        [r["iteration"] + 600 for r in tr], [r["energy_pJ"] / 1000 for r in tr], lw=2
    )
    plt.axhspan(0.1, 0.5, color="#23947b", alpha=0.12)
    plt.xlabel("累计真实圈数（从原第600圈继续）")
    plt.ylabel("输出能量 / nJ")
    plt.title("已有候选的80圈新求解器复算")
    plt.grid(alpha=0.2)
    html.append(save("adaptive_candidate_energy"))
    html.append(
        f'<p>OC→CNT、OC30%、GDD+0.2ps²、pump20mW：继续80个真实往返，窗口内最大峰数为{summary["peaks_max"]}；能量CV为{summary["energy_cv"]:.1%}，有效增益年龄仅{summary["relaxation_age"]:.4f}，远小于5。因此分类为未收敛，不能因为单峰就通过。</p>'
    )
    html.append(
        f"<p>1mW低泵浦负对照：弱种子在64个直接验证圈中衰减；这一有限记录不用于宣称全参数阈值。两项物理pilot没有满足受限增益加速的接受条件，不能据此声称已验证实际加速倍数。</p>"
    )
    html.append(
        f'<p>路径集成检查：泵浦双向{paths["pump_path_points"]}点、GDD双向{paths["gdd_path_points"]}点、多种子{paths["seed_cases"]}组；仅运行2–4圈验证状态继承和标签，不是物理吸引子搜索结果。12项数值单元测试另覆盖能量守恒、回退不重复增益、EDF重算、泵浦初态、解析增益步、偏振周期二以及反转均值抵消反例。</p>'
    )
    html.append(
        "<p><b>适用边界：</b>当前新路径为CPU精度参考实现，旧GPU地图代码不自动获得这些功能。Floquet、多时间槽共享增益与卫星竞争、RK4IP及GPU并行前缀尚未实现；这些缺项会限制最终稳定性认证，但不阻止使用新算法继续搜索。</p>"
    )
    fragment = "".join(html)
    (ROOT / "adaptive_algorithm_validation.html").write_text(fragment, encoding="utf8")
    page = (
        '<meta charset="utf-8"><title>自适应求解器改造与验证</title><style>body{font-family:system-ui;max-width:1120px;margin:35px auto;padding:20px;line-height:1.7;color:#203648}img{display:block;margin:24px 0}h1{color:#086887}</style><h1>自适应求解器改造与验证</h1>'
        + fragment
    )
    (ROOT / "adaptive_solver_report.html").write_text(page, encoding="utf8")
    print(ROOT / "adaptive_solver_report.html")


if __name__ == "__main__":
    main()
