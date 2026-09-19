"""Standalone Chinese report, generated only from saved simulation results."""

from pathlib import Path
import base64, csv, html, json, hashlib
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from config import ROOT, OC, PUMPS, GDD, TOPOLOGIES, config

F = ROOT / "figures"
F.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 12,
    }
)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


coarse = sum([read(ROOT / f"group_{g:02d}.json") for g in range(10)], [])
best = []
datasets = {}
for g in range(10):
    p = ROOT / f"fine_group_{g:02d}.json"
    if not p.exists():
        p = ROOT / f"group_{g:02d}.json"
    rows = read(p)
    for r in rows:
        r["source"] = p.with_suffix(".mat").name
        r["dt_ps"] = 0.125 if p.stem.startswith("fine") else 0.5
    best.extend(rows)
    datasets[g] = (rows, p.with_suffix(".mat"))
longfiles = sorted(ROOT.glob("long_g??_c??.json"))
longrows = sum([read(p) for p in longfiles], [])


def long_class(r):
    if r["status"] != "completed":
        return "长程数值边界未通过"
    if not r["single_tail"]:
        return "末段未保持单峰"
    if not (r["energy_min_pJ"] >= 100 and r["energy_max_pJ"] <= 500):
        return "末段能量离开目标范围"
    if r["energy_cv"] < 1e-4 and r["residual"] < 1e-3:
        return "末段逐圈重复候选"
    return "末段目标单峰；波形重复性未通过"


late_targets = [
    r
    for r in longrows
    if long_class(r) in ["末段逐圈重复候选", "末段目标单峰；波形重复性未通过"]
]


def table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join("<th>" + html.escape(str(x)) + "</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join("<td>" + html.escape(str(x)) + "</td>" for x in row)
            + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def savefig(name):
    plt.tight_layout()
    plt.savefig(F / f"{name}.png", dpi=150)
    plt.savefig(F / f"{name}.svg")
    plt.close()
    return (
        '<img loading="lazy" alt="'
        + name
        + '" src="data:image/png;base64,'
        + base64.b64encode((F / f"{name}.png").read_bytes()).decode()
        + '">'
    )


def topo(x):
    return "OC → CNT" if x == "OC_CNT" else "CNT → OC"


stop_labels = {
    "completed": "完成规定圈数",
    "spectral_boundary": "频谱边界越界",
    "time_boundary": "时间窗边界越界",
    "decayed_or_nonfinite": "衰减或非有限数值",
}
statusnames = [
    "数值边界未通过",
    "能量不在目标内",
    "目标能量但多峰",
    "目标单峰但未稳定",
    "通过短时初筛",
]
colors = ["#d9dfe6", "#edf2f6", "#a891c7", "#f3b56c", "#279b87"]


def status(r):
    if not r["numerical_pass"]:
        return 0
    if not r["energy_range_pass"]:
        return 1
    if not r["single_tail"]:
        return 2
    if not r["screen_candidate"]:
        return 3
    return 4


sections = []


def section(title, body):
    sections.append((title, body))


valid = [r for r in best if r["numerical_pass"]]
target = [r for r in valid if r["energy_range_pass"]]
single = [r for r in target if r["single_tail"]]
candidates = [r for r in best if r["screen_candidate"]]
section(
    "研究目标与本轮结果",
    f"""<p class="eyebrow">原生约 10 MHz · 固定 20.42 m · 双拓扑 · 640 个不同参数组合</p>
<h2>单脉冲 0.1–0.5 nJ 的色散／能量地图</h2>
<p>目标等价于约 1–5 mW 平均输出。扫描 OC 输出比例 20–90%、净群延迟色散 −0.1 至 +0.3 ps²、976 nm 泵浦 5–150 mW。两种器件顺序使用相同材料参数与成对初态。</p>
<div class="cards"><div><b>640</b>完整乘积网格</div><div><b>{len(valid)}</b>通过短时边界检查</div><div><b>{len(single)}</b>目标能量且连续单峰</div><div><b>{len(candidates)}</b>通过全部短时初筛门槛</div></div>
<p class="callout">{'出现短时初筛候选，仍须长程及网格收敛验证。' if candidates else '当前网格尚未确认稳定单脉冲可行区。已有目标能量附近的状态，但没有同时通过全部短时稳定性门槛；这不等于证明该架构无法实现 0.1–0.5 nJ。'}</p>
<p>这里“完成扫描”表示每个参数点均已推进或因明确的数值边界门槛停止。灰色区域仍属未解析区域，不能当作物理不可行区。所有曲线均为数值仿真，没有实物实验数据。</p>""",
)
lengthrows = []
for gd in GDD:
    c = config("OC_CNT", gd)
    s = c["segments"]
    lengthrows.append(
        [
            f"{gd:+.2f}",
            f"{s[1,0]:.4f}",
            f"{s[[0,2,3,4],0].sum():.4f}",
            f"{s[5,0]:.4f}",
            f"{s[:,0].sum():.5f}",
        ]
    )
section(
    "固定总腔长与两种器件顺序",
    "<p>模型沿传播方向为：0.10 m SMF → 1.50 m EDF → 0.05 m SMF → 0.72 m SMF → PC → <b>OC／CNT 相邻器件</b> → 其余 SMF → NDF → 混合器件损耗 → 回到起点。只交换 OC 与 CNT；PC、接续损耗、光纤段位置保持不变。</p>"
    + '<div class="routes"><p><b>顺序 A</b>　PC → OC（输出）→ CNT → 接续损耗</p><p><b>顺序 B</b>　PC → CNT → OC（输出）→ 接续损耗</p></div>'
    + "<p>EDF 固定 1.5 m。用 NDF 替换等长 SMF 调节净色散，始终满足 L<sub>EDF</sub> + L<sub>SMF</sub> + L<sub>NDF</sub> = 20.42 m。共用群折射率 1.4682，计算重频约 9.999534 MHz；未计不同纤种群折射率差与器件内部光程，实物需据实微调。</p>"
    + '<p class="formula">GDD = 0.0459 L<sub>EDF</sub> − 0.0217 L<sub>SMF</sub> + 0.1000 L<sub>NDF</sub>　[ps²]</p>'
    + table(
        ["净 GDD / ps²", "EDF / m", "SMF 合计 / m", "NDF / m", "总长 / m"], lengthrows
    )
    + "<p>NDF 的 β₂ = +0.10 ps²/m、γ = 0.004 W⁻¹m⁻¹ 是沿用的模型假设；可以代表一类正常色散补偿光纤，不能直接当作任意国产 DCF 型号的已核实规格。此处扫描的是净 β₂L，不是 D 参数。</p>",
)
plt.figure(figsize=(12, 5.8))
for gd in GDD:
    s = config("OC_CNT", gd)["segments"]
    plt.plot(
        np.r_[0, np.cumsum(s[:, 0])],
        np.r_[0, np.cumsum(s[:, 0] * s[:, 1])],
        marker="o",
        label=f"净 GDD {gd:+.1f} ps²",
    )
plt.xlabel("沿腔传播位置 / m")
plt.ylabel("累积群延迟色散 / ps²")
plt.title("固定总长条件下，各段光纤的色散累积")
plt.grid(alpha=0.2)
plt.legend()
section(
    "分段色散的空间分布",
    savefig("cumulative_dispersion")
    + "<p>前端 EDF 和短尾纤不变，长 SMF 段先积累负色散，末端 NDF 再补偿至指定净值。因而“净色散相同”不代表腔内任一点脉宽相同；OC 与 CNT 位于长 SMF／NDF 之前，输出脉冲取自实际 OC 位置。</p>",
)
c = config("OC_CNT", 0.2)
section(
    "材料、损耗与动态模型",
    table(
        ["项目", "本轮计算设置", "证据范围"],
        [
            [
                "EDF",
                "1.5 m；β₂=+0.0459 ps²/m；γ=0.00496 W⁻¹m⁻¹",
                "本轮实际使用 segments 数组，不使用旧配置同名标量",
            ],
            ["SMF", "β₂=−0.0217 ps²/m；γ=0.0013 W⁻¹m⁻¹", "沿用材料模型"],
            [
                "泵浦",
                "976 nm；5/10/15/20/30/50/100/150 mW；入腔路径损耗0.6 dB",
                "标注为泵 LD 输出功率，不是 EDF 已吸收功率",
            ],
            [
                "EDF 增益",
                "αp=7.5 m⁻¹；αs=2.17594 m⁻¹；发射系数2.82873 m⁻¹；寿命10 ms；增益带宽30 nm",
                "有效参数，未对具体实物联合标定",
            ],
            [
                "CNT",
                "调制深度5%；非饱和损耗10%；恢复时间0.5 ps；饱和能量20 pJ",
                "动态吸收模型；等效饱和功率40 W；本轮未偷偷改为1 pJ",
            ],
            [
                "额外损耗",
                "OC 0.15 dB；接续0.5 dB；混合器件1.1 dB，PDL 0.15 dB",
                "仅实际传播代码中的损耗项计入",
            ],
            [
                "偏振",
                "双偏振矢量场、固定双折射及PC、弱PDL",
                "不是全保偏，也未扫描PC角度",
            ],
            [
                "其他",
                "保留三阶色散、Kerr 非线性；无额外窄带滤波；ASE关闭",
                "不证明噪声自启动或100 ns全周期内无其他脉冲",
            ],
        ],
    )
    + '<p class="formula">dq/dt = (q₀−q)/τ<sub>SA</sub> − qP/E<sub>sat</sub>；T<sub>CNT</sub> = 1−l<sub>ns</sub>−q</p>'
    + "<p>CNT 吸收随局部瞬时功率降低，且在脉冲之间恢复。OC→CNT 中，CNT 接收的是耦出后的剩余光；CNT→OC 中，CNT 先承受全部入射光。高 OC 下差异尤其明显，但更强漂白并不自动保证单脉冲稳定。</p>"
    + '<p class="formula">dn/dt = A−Bn；n<sub>下一圈</sub> = A/B + (n−A/B) exp(−B T<sub>R</sub>)</p>'
    + "<p>n 为 EDF 各段反转粒子数比例；A、B 由沿程泵浦、信号受激跃迁和寿命构成，每圈更新一次。保留增益储能，不把每一圈强制设成静态平衡；因此早期落入能量区间可能只是增益恢复或耗尽过程。</p>"
    + '<p><a download="effective_parameters.json" href="data:application/json;base64,'
    + base64.b64encode(
        (
            Path(__file__).resolve().parent / "parameters/effective_parameters.json"
        ).read_bytes()
    ).decode()
    + '">下载有效模型参数（含各段光纤、双折射及PC设置）</a></p>',
)
section(
    "扫描设计、初始化与判据",
    table(
        ["维度", "取值"],
        [
            ["OC 输出比例", "20/30/40/50/60/70/80/90%"],
            ["净 GDD", "−0.1 / 0 / +0.1 / +0.2 / +0.3 ps²"],
            ["泵浦", "5/10/15/20/30/50/100/150 mW"],
            ["器件顺序", "OC→CNT；CNT→OC"],
            ["总计", "2 × 8 × 5 × 8 = 640 个不同参数组合"],
        ],
    )
    + "<p>每个 OC 使用近似小信号净增益为零的均匀反转初值；初始高斯强度 σ=10 ps，能量由约0.3 nJ输出的猜测换算。<b>只初始化一次，传播中无能量归一化。</b>成对拓扑共用相同场与反转初值。这个初态是搜索起点，不是已建立的锁模解，也不是泵浦稳态。</p>"
    + "<p>初筛最多600圈，约60 μs。粗网格 Δt=0.5 ps、2048点；细网格 Δt=0.125 ps、8192点，时间窗均1024 ps，保持相同物理参数和初态复算。EDF 最大步长0.1 m、无源段0.5 m；延长对照使用Δt=0.25 ps、无源段0.25 m。输出场外侧各5%的时间／频谱能量用于越界检查，每20圈检查一次。</p>"
    + table(
        ["层级", "操作定义", "含义"],
        [
            [
                "数值边界",
                "时间边缘占比<10⁻⁶，频谱边缘占比<10⁻⁸，且完成600圈",
                "只是输出面边界检查通过，不等于完整网格收敛",
            ],
            [
                "目标能量",
                "最后100圈每圈能量均在100–500 pJ",
                "不用单圈或平均值代替范围约束",
            ],
            [
                "单峰",
                "最后64圈显著峰数均为1；峰高和突起均≥最高峰10%",
                "窗口内单主峰初筛；弱卫星和全周期其他脉冲未排除",
            ],
            [
                "短时稳定",
                "末100圈能量CV<10⁻⁴；末64圈最大对齐强度残差<10⁻³",
                "需同时成立；残差允许整体时间平移，不移除形状／能量变化",
            ],
            [
                "模型内可行区",
                "以上通过，并经长程、细步长、多初态及反转收敛复核",
                "本轮未确认；不能因一个快照满足目标而放行",
            ],
        ],
    )
    + "<p>上述稳定性门槛用于检查“逐圈重复的固定波形”，不是通用锁模定义。未过门槛不自动等于未锁模，也不排除有界单脉冲呼吸态；后者需长记录确认周期性、能量范围及全周期脉冲数。本轮没有将短暂单峰直接视为稳定呼吸态。</p>",
)

maps = {}
for group, (rows, path) in datasets.items():
    label = f'{topo(rows[0]["topology"])}；净 GDD {rows[0]["GDD_ps2"]:+.1f} ps²'
    for metric in ["energy", "status", "peaks", "cv"]:
        fig, ax = plt.subplots(figsize=(11.6, 7))
        arr = np.full((8, 8), np.nan)
        for r in rows:
            x = r["index"] // 8
            y = r["index"] % 8
            if metric == "status":
                arr[y, x] = status(r)
            elif r["numerical_pass"]:
                arr[y, x] = r[
                    {"energy": "energy_mean_pJ", "peaks": "peaks", "cv": "energy_cv"}[
                        metric
                    ]
                ] / (1000 if metric == "energy" else 1)
        if metric == "status":
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(np.arange(-0.5, 5.5), 5)
            im = ax.imshow(arr, origin="lower", cmap=cmap, norm=norm, aspect="auto")
            bar = fig.colorbar(im, ax=ax, ticks=range(5), pad=0.025)
            bar.ax.set_yticklabels(statusnames)
        else:
            cmap = plt.get_cmap("viridis" if metric == "energy" else "magma").copy()
            cmap.set_bad("#d9dfe6")
            norm = (
                LogNorm(0.001, 10)
                if metric == "energy"
                else LogNorm(1e-4, 3) if metric == "cv" else None
            )
            im = ax.imshow(
                arr,
                origin="lower",
                cmap=cmap,
                norm=norm,
                aspect="auto",
                **({"vmin": 1, "vmax": 20} if metric == "peaks" else {}),
            )
            bar = fig.colorbar(im, ax=ax, pad=0.025)
            bar.set_label(
                {
                    "energy": "末100圈平均输出能量 / nJ",
                    "peaks": "末圈显著峰数",
                    "cv": "末100圈能量CV（标准差/均值）",
                }[metric]
            )
        for r in rows:
            x = r["index"] // 8
            y = r["index"] % 8
            v = arr[y, x]
            txt = (
                str(int(v))
                if metric in ["status", "peaks"] and np.isfinite(v)
                else (
                    f"{v:.3f}"
                    if metric == "energy" and np.isfinite(v) and v >= 0.001
                    else f"{v:.1e}" if np.isfinite(v) else "×"
                )
            )
            ax.text(
                x,
                y,
                txt,
                ha="center",
                va="center",
                fontsize=10,
                color="black",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.4),
            )
            if metric == "energy" and np.isfinite(v) and 0.1 <= v <= 0.5:
                ax.add_patch(
                    Rectangle(
                        (x - 0.48, y - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        lw=2.5,
                        ec="#ef7a2d",
                    )
                )
        ax.set_xticks(range(8), [f"{v:.0%}" for v in OC])
        ax.set_yticks(range(8), [f"{v*1000:g}" for v in PUMPS])
        ax.set_xlabel("输出耦合比例 OC")
        ax.set_ylabel("976 nm 泵 LD 输出功率 / mW（离散扫描点）")
        ax.set_title(
            label
            + "\n"
            + {
                "energy": "输出能量地图",
                "status": "分层筛选状态",
                "peaks": "输出脉冲的显著峰数",
                "cv": "输出能量随圈数的相对波动",
            }[metric],
            pad=14,
        )
        maps[f"{group}_{metric}"] = savefig(f"map_{group:02d}_{metric}")
maphtml = (
    '<div class="controls"><label>器件顺序 <select id="topo"><option value="0">OC → CNT</option><option value="1">CNT → OC</option></select></label><label>净 GDD <select id="gdd">'
    + "".join(
        f'<option value="{i}" {"selected" if i==3 else ""}>{g:+.1f} ps²</option>'
        for i, g in enumerate(GDD)
    )
    + '</select></label><label>显示量 <select id="metric"><option value="energy">输出能量</option><option value="status">筛选状态</option><option value="peaks">显著峰数</option><option value="cv">能量波动 CV</option></select></label></div>'
)
maphtml += "".join(
    f'<div class="map" data-map="{key}" hidden>{val}</div>' for key, val in maps.items()
)
section(
    "OC × 泵浦 × 净色散：交互地图",
    maphtml
    + "<p>每个方格是一组已计算工况，不对格点之间插值。能量图橙框仅表示<b>最后100圈的均值</b>位于0.1–0.5 nJ；是否每圈都在目标内、是否单峰与稳定，须查看筛选状态。灰色×为边界检查失败，不显示失效时刻的能量，以免误导。</p>"
    + "<p>状态代码：0 数值未解析；1 能量范围未满足；2 目标能量但未保持单峰；3 目标能量且单峰但稳定性不足；4 通过短时初筛。状态1不表示其它条件已通过。峰数图显示末圈，最终单峰判据检查末64圈。</p>",
)
summaryrows = []
comparisons = []
for g, (rows, _) in datasets.items():
    cr = [r for r in coarse if r["group"] == g]
    common = [
        abs(x["energy_pJ"] - y["energy_pJ"]) / max(abs(y["energy_pJ"]), 1e-100)
        for x, y in zip(cr, rows)
        if x["numerical_pass"] and y["numerical_pass"]
    ]
    comparisons.append(
        dict(
            group=g,
            coarse_valid=sum(r["numerical_pass"] for r in cr),
            display_valid=sum(r["numerical_pass"] for r in rows),
            dt_ps=rows[0]["dt_ps"],
            common=len(common),
            median_relative_energy=float(np.median(common)) if common else None,
            max_relative_energy=max(common) if common else None,
        )
    )
    summaryrows.append(
        [
            topo(rows[0]["topology"]),
            f'{rows[0]["GDD_ps2"]:+.1f}',
            f'{rows[0]["dt_ps"]:g}',
            sum(r["numerical_pass"] for r in cr),
            sum(r["numerical_pass"] for r in rows),
            sum(r["numerical_pass"] and r["energy_range_pass"] for r in rows),
            sum(status(r) == 3 for r in rows),
            sum(r["screen_candidate"] for r in rows),
        ]
    )
(ROOT / "grid_comparison.json").write_text(
    json.dumps(comparisons, indent=2), encoding="utf8"
)
section(
    "各色散与拓扑的结果汇总",
    table(
        [
            "顺序",
            "GDD / ps²",
            "展示Δt / ps",
            "粗网格边界通过 /64",
            "展示网格边界通过 /64",
            "每圈目标能量",
            "目标单峰但未稳定",
            "全部短时门槛",
        ],
        summaryrows,
    )
    + "<p>加密结果替换同一格点的粗结果用于展示，两份原始数据均保留。时间采样细化不能替代全程空间步长与时间窗收敛；不能把地图间细微差异都当成可靠的物理差异。特别是仍灰色的格点，还需要更宽频谱范围、空间步长收敛或更宽时间窗来判别。</p>",
)
shown = sorted(single, key=lambda r: r["energy_cv"])[:8]
section(
    "目标范围内的单峰状态与稳定性差距",
    table(
        [
            "编号",
            "顺序",
            "OC",
            "GDD",
            "pump/mW",
            "末100圈能量范围/nJ",
            "能量CV",
            "最大强度残差",
            "末圈FWHM/ps",
        ],
        [
            [
                r["id"],
                topo(r["topology"]),
                f'{r["oc"]:.0%}',
                f'{r["GDD_ps2"]:+.1f}',
                r["pump_mW"],
                f'{r["energy_min_pJ"]/1000:.3f}–{r["energy_max_pJ"]/1000:.3f}',
                f'{r["energy_cv"]:.2%}',
                f'{r["residual"]:.2%}',
                r["FWHM_ps"],
            ]
            for r in shown
        ],
    )
    + "<p>这里列的是值得追踪的状态，不是已接受的工作点。CV表示时间演化的相对变化，不是蒙特卡洛不确定度；FWHM是末圈主峰半高宽，不能代替稳定输出脉宽。</p>",
)
plt.figure(figsize=(12, 5.5))
for g, idx in [(2, 41), (3, 52), (8, 52)]:
    m = loadmat(ROOT / f"group_{g:02d}.mat", simplify_cells=True)
    tr = m["trace"][: int(m["completed"][idx]), idx]
    plt.plot(
        np.arange(1, len(tr) + 1) / 10,
        tr[:, 0] / 1000,
        label=f'{topo(m["c"]["topology"])} / {m["c"]["map_GDD_ps2"]:+.1f} ps² / OC {m["oc"][idx]:.0%} / {m["pumps"][idx]*1000:g} mW',
    )
plt.axhspan(0.1, 0.5, alpha=0.13, color="#279b87", label="目标0.1–0.5 nJ")
plt.yscale("log")
plt.xlabel("演化时间 / μs（约10圈/μs）")
plt.ylabel("OC 输出能量 / nJ")
plt.title("接近目标的代表状态：能量随演化时间的变化")
plt.grid(alpha=0.2)
plt.legend(fontsize=10)
section(
    "代表状态的能量演化",
    savefig("representative_energy")
    + "<p>保留从初态到600圈的完整历史。能量穿过目标带而没有停留，是未收敛的直接证据；不能只截取目标带内的一圈作为锁模输出结果。</p>",
)
for g in [3, 8]:
    m = loadmat(ROOT / f"group_{g:02d}.mat", simplify_cells=True)
    idx = 52
    p = np.sum(abs(m["out"][idx]) ** 2, axis=0)
    t = m["t"]
    center = t[np.argmax(p)]
    plt.figure(figsize=(12, 5.5))
    plt.plot(t - center, p, color="#087fa7" if g == 3 else "#ad5e28", lw=2)
    plt.xlim(-150, 150)
    plt.xlabel("相对主峰时间 / ps")
    plt.ylabel("瞬时输出功率 / W")
    plt.title(f'{topo(m["c"]["topology"])}：+0.2 ps²、OC 80%、pump 30 mW，第600圈快照')
    plt.grid(alpha=0.2)
    section(
        f'{topo(m["c"]["topology"])} 的输出脉冲快照',
        savefig(f"pulse_g{g:02d}")
        + "<p>两图对应相同材料、OC、色散与泵浦设置。它们是实际输出场快照，尚未获得稳定性认证；总能量和主峰形状必须结合前后演化一起看。</p>",
    )
if longfiles:
    section(
        "延长演化与初始能量扰动",
        table(
            [
                "顺序",
                "GDD / OC / pump",
                "初态能量倍率",
                "新增圈数",
                "停止状态",
                "末500圈能量范围/nJ",
                "能量CV",
                "判读",
            ],
            [
                [
                    topo(TOPOLOGIES[r["group"] // 5]),
                    f'{GDD[r["group"]%5]:+.1f} ps² / {OC[r["index"]//8]:.0%} / {PUMPS[r["index"]%8]*1000:g} mW',
                    r["factor"],
                    r["rounds"],
                    stop_labels[r["status"]],
                    f'{r["energy_min_pJ"]/1000:.4g}–{r["energy_max_pJ"]/1000:.4g}',
                    f'{r["energy_cv"]:.2%}',
                    long_class(r),
                ]
                for r in longrows
            ],
        )
        + "<p>从各组第600圈内部场继续，能量分别乘0.9、1.0、1.1；反转与CNT状态保持继承。时间步长0.125或0.25 ps，不比源状态更粗；无源空间步长0.25 m。c52对应OC80%/pump30 mW；c09对应OC30%/pump10 mW；c11对应OC30%/pump20 mW。这里只检验已有脉冲状态的延续，不等价于独立冷启动。</p>"
        + f"<p>目前有{len(late_targets)}条延续轨迹在最后500圈同时保持目标能量和单主峰。若波形重复性未通过，只能列为末段目标单峰状态；确认呼吸态还需检查多周期重复性、场分布与更长记录，不能仅凭能量周期性命名。</p>",
    )
    for path in longfiles:
        m = loadmat(path.with_suffix(".mat"), simplify_cells=True)
        g = int(m["source_group"])
        plt.figure(figsize=(12, 5.5))
        for i, factor in enumerate(m["factors"]):
            steps = int(m["completed"][i])
            tr = m["trace"][:steps, i]
            plt.plot(
                (np.arange(steps) + 601) / 10,
                tr[:, 0] / 1000,
                label=f"初始能量×{factor:g}",
                lw=1.4,
            )
        index = int(m["source_index"])
        description = f"{topo(TOPOLOGIES[g//5])} / {GDD[g%5]:+.1f} ps² / OC {OC[index//8]:.0%} / pump {PUMPS[index%8]*1000:g} mW"
        plt.axhspan(0.1, 0.5, color="#279b87", alpha=0.13)
        plt.yscale("log")
        plt.xlabel("从主扫描初态起的累计演化时间 / μs")
        plt.ylabel("输出能量 / nJ")
        plt.title(description + "\n目标附近状态的延续与扰动响应")
        plt.grid(alpha=0.2)
        plt.legend()
        section(
            description + "：长程输出",
            savefig(path.stem + "_energy")
            + "<p>曲线结束于实际完成圈数；触发边界停止时，后续状态未知。不能将停止以后的能量外推为恒定输出。</p>",
        )
        if index == 9:
            plt.figure(figsize=(12, 5.5))
            for i, factor in enumerate(m["factors"]):
                steps = int(m["completed"][i])
                tr = m["trace"][:steps, i]
                plt.plot(
                    (np.arange(steps) + 601) / 10,
                    tr[:, 5],
                    label=f"初始能量×{factor:g}",
                    lw=1.3,
                )
            plt.xlabel("从主扫描初态起的累计演化时间 / μs")
            plt.ylabel("EDF各段平均反转比例")
            plt.title(description + "\n与输出能量同步记录的增益状态")
            plt.grid(alpha=0.2)
            plt.legend()
            section(
                "低泵浦目标点：EDF反转的动态记录",
                savefig(path.stem + "_inversion")
                + "<p>反转比例表示增益介质处于激发态的粒子比例，按EDF等长单元取平均。该图与前面的能量曲线使用相同时间轴，用于检查储能是否仍在变化；平均反转近似不变也不能单独证明每段反转及脉冲形状已收敛。</p>",
            )
            field = m["out"][1]
            p = np.sum(abs(field) ** 2, axis=0)
            dt = float(m["dt"])
            t = m["t"]
            center = t[np.argmax(p)]
            energy = float(p.sum() * dt / 1000)
            peaks = len(
                find_peaks(p, height=0.1 * p.max(), prominence=0.1 * p.max())[0]
            )
            last = int(m["completed"][1]) + 600
            label = f"第{last}圈，输出{energy:.4f} nJ；全窗显著峰数{peaks}"
            plt.figure(figsize=(12, 5.5))
            plt.plot(t - center, p, lw=1.8)
            plt.xlim(t[0] - center, t[-1] - center)
            plt.xlabel("相对主峰时间 / ps")
            plt.ylabel("瞬时输出功率 / W")
            plt.title(description + "\n" + label)
            plt.grid(alpha=0.2)
            stopped = read(path)[1]["status"] != "completed"
            diagnostic = (
                '<p class="callout">该末圈已触发数值边界门槛，以下仅作为停止时的诊断快照，不能作为已验证的光源输出参数。</p>'
                if stopped
                else ""
            )
            section(
                "低泵浦目标点：末圈时域诊断",
                diagnostic
                + savefig(path.stem + "_pulse")
                + "<p>使用未扰动分支的实际末圈场，展示完整1024 ps时间窗，原点移至主峰。是否保持单脉冲及目标能量，以全窗峰数和长程表为准；此图不单独构成稳定性证明。</p>",
            )
            spec = np.fft.fftshift(np.sum(abs(np.fft.fft(field, axis=-1)) ** 2, axis=0))
            freq = np.fft.fftshift(np.fft.fftfreq(len(t), dt))
            db = 10 * np.log10(np.maximum(spec / spec.max(), 1e-12))
            extent = max(0.05, float(np.max(abs(freq[db > -60]))) * 1.15)
            plt.figure(figsize=(12, 5.5))
            plt.plot(freq, db, lw=1.8)
            plt.xlim(-extent, extent)
            plt.ylim(-80, 3)
            plt.xlabel("相对光学载频的频率偏移 / THz")
            plt.ylabel("输出场光谱包络 / dB（相对峰值）")
            plt.title(description + "\n末圈局部时间窗内的输出场光谱包络")
            plt.grid(alpha=0.2)
            section(
                "低泵浦目标点：末圈光谱诊断",
                diagnostic
                + savefig(path.stem + "_spectrum")
                + "<p>这是局部时间窗内输出场的傅里叶包络，未解析10 MHz间隔的光梳线。光谱与时域图来自同一输出场；不把该包络当作已验证的射频谱或光梳线宽。</p>",
            )
if (ROOT / "best_candidate_validation.html").exists():
    section(
        "最低短时波动候选：延长演化、时间窗与网格验证",
        (ROOT / "best_candidate_validation.html").read_text(encoding="utf8"),
    )
budget = read(ROOT / "steady_energy_budget.json")
budgettable = []
for fraction in OC:
    values = []
    for topology in TOPOLOGIES:
        for pumpmw in [10, 15]:
            energies = [
                r["energy_nJ"]
                for r in budget["rows"]
                if r["topology"] == topology
                and r["oc"] == fraction
                and r["pump_mW"] == pumpmw
            ]
            values.append(f"{min(energies):.3f}–{max(energies):.3f}")
    budgettable.append([f"{fraction:.0%}"] + values)
section(
    "目标能量附近的稳态泵浦预算",
    table(
        ["OC", "OC→CNT，10mW", "OC→CNT，15mW", "CNT→OC，10mW", "CNT→OC，15mW"],
        budgettable,
    )
    + "<p>表内单位为nJ，范围来自CNT透过率85–90%及PDL0–0.15 dB的两端假设。在当前材料参数下，10–15 mW泵浦在多数OC设置下具有目标能量量级；泵浦增加到30 mW并不一定适合0.1–0.5 nJ目标。<b>这是中心频率稳态预算，不是已验证的单脉冲可行区。</b>实际脉冲还受光谱、色散、非线性和动态稳定性影响。</p>",
)
for topology in TOPOLOGIES:
    lo = [
        r
        for r in budget["rows"]
        if r["topology"] == topology and r["bound"] == "loss_high"
    ]
    hi = [
        r
        for r in budget["rows"]
        if r["topology"] == topology and r["bound"] == "loss_low"
    ]
    plt.figure(figsize=(12, 6))
    for fraction in [0.2, 0.5, 0.8, 0.9]:
        low = np.array([r["energy_nJ"] for r in lo if r["oc"] == fraction])
        high = np.array([r["energy_nJ"] for r in hi if r["oc"] == fraction])
        low = np.where(low > 0, low, np.nan)
        high = np.where(high > 0, high, np.nan)
        line = plt.plot(PUMPS * 1000, low, marker="o", label=f"OC {fraction:.0%}")[0]
        plt.fill_between(PUMPS * 1000, low, high, color=line.get_color(), alpha=0.13)
    plt.axhspan(0.1, 0.5, color="#279b87", alpha=0.08, label="目标能量带")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(0.01, 10)
    plt.xlabel("976 nm 泵 LD 输出功率 / mW")
    plt.ylabel("中心频率稳态能量预算 / nJ")
    plt.title(topo(topology) + "：独立稳态能量预算（不含锁模稳定性）")
    plt.grid(alpha=0.2)
    plt.legend()
    section(
        topo(topology) + " 的稳态能量预算",
        savefig("budget_" + topology)
        + "<p>独立求解EDF沿程泵浦／信号功率、局部稳态反转及闭环增益=损耗。实线按CNT透过率85%、PDL0.15 dB；色带另一端按CNT透过率90%、PDL0 dB。只表示这两项损耗变化下的预算范围，不是置信区间，也不是实际脉冲的严格上下界。</p>"
        + "<p>低于该简化模型振荡阈值的零输出点不画入对数坐标。此模型把信号集中在增益中心、忽略脉冲光谱与稳定性，所以不随净GDD变化。它可辅助判断泵浦能量量级，不替代前面的动态地图。预算进入目标带而脉冲未稳定，说明需要继续解决脉冲选择和增益动力学，而不能只增加泵浦。</p>",
    )
plt.figure(figsize=(12, 5.5))
for g in [2, 7]:
    m = loadmat(ROOT / f"group_{g:02d}.mat", simplify_cells=True)
    values = m["trace"][0, np.arange(8) * 8 + 1, 8]
    plt.plot(OC * 100, values, marker="o", lw=2, label=topo(TOPOLOGIES[g // 5]))
plt.axhline(40, color="#955025", ls="--", label="Esat/τ = 40 W（模型尺度）")
plt.xlabel("输出耦合比例 OC / %")
plt.ylabel("第一次通过CNT的峰值功率 / W")
plt.title("成对相同初态下，器件顺序对CNT入射功率的直接影响")
plt.grid(alpha=0.2)
plt.legend()
section(
    "CNT处峰值功率的成对对照",
    savefig("CNT_first_pass_power")
    + "<p>固定+0.1 ps²、10 mW泵浦，取首次通过CNT的局部峰值。两种顺序在该位置之前具有相同初场，差别是OC→CNT已先耦出能量。虚线40 W来自本轮Esat/τ，并非器件损伤阈值，也不是锁模的硬门槛。该图验证直接功率作用；600圈后的差异还包含脉形与增益反馈。</p>",
)
section(
    "OC 与 CNT 顺序如何影响工作区",
    "<p>OC→CNT：先耦出，再让剩余光通过CNT。高输出耦合减少回腔能量，也降低CNT处峰值功率，可能减弱可饱和吸收的脉冲选择作用。</p>"
    + "<p>CNT→OC：CNT先承受放大后的光，可饱和吸收更强，但输出需经过CNT损耗，腔内非线性、增益耗尽及脉冲分裂也会改变。因此不能仅凭“更容易漂白”认定这一顺序更优。</p>"
    + "<p>本轮成对网格控制了光纤、损耗、PC和初态，差异来自器件顺序与随后的非线性反馈。尚未确认稳定工作点，所以当前不能按末圈功率给两种拓扑作采购优劣排序。</p>",
)
section(
    "数值验证与结论边界",
    table(
        ["验证", "实际结果"],
        [
            ["CPU批量/标量传播", "6组成对参数，10圈，最大相对场误差约6.61×10⁻¹⁵"],
            [
                "GPU/CPU传播",
                "6组本轮初始化工况，10圈，融合内核版本最大相对场误差约2.33×10⁻¹²",
            ],
            [
                "局部网格复核",
                "3个目标附近状态延续20圈，Δt/无源步长由0.25/0.25减至0.125/0.125；能量相对差≤7.05×10⁻⁷、强度差≤1.59×10⁻⁵",
            ],
            ["频谱细化", "保持物理参数重算；没有修改CNT、泵浦或损耗来凑目标"],
            [
                "稳态预算",
                "EDF积分150→300步，输出相对差<3.4×10⁻⁸；满足泵浦/信号光子转换上限",
            ],
            [
                "未完成认证",
                "全腔各段频谱/空间步长收敛、100 ns全周期、多噪声冷启动、实际材料参数标定",
            ],
        ],
    )
    + "<p>上述交叉验证检查同一离散模型的实现一致性，不等于验证模型与实物一致。600圈约60 μs，远短于10 ms材料寿命；泵浦下有效反转响应可更快，但仍须用反转收敛和长程历史判断，不能只用固定圈数宣称稳定。</p>"
    + "<p><b>本轮可用结论：</b>两种顺序都已进行所列离散网格的完整OC×GDD×pump搜索；目标能量并非从未出现，但当前证据尚不足以冻结稳定单脉冲设计。下一轮应围绕已解析的目标附近点检查增益平衡和CNT漂白，并先解决灰色区域的数值分辨率问题；不要把整个灰色区域删掉或宣布不可行。</p>"
    + "<p>本轮只有五个净色散截面，并未穷尽连续参数空间；尤其未覆盖零色散附近更细的间隔和不同初始啁啾，不能据此排除狭窄的稳定区。PC与CNT参数固定，结论也不能外推到其它器件参数。</p>",
)
section(
    "完整参数表与数据导出",
    '<p>下表保留640个不同参数组合，可筛选顺序、色散或编号。能量单位为nJ，失败工况标为“未解析”；原始MAT中保留停止前全部逐圈诊断。</p><input id="search" placeholder="筛选，例如 OC_CNT、g03 或 +0.2" aria-label="筛选参数表"><button id="download">下载完整 CSV</button><div id="data-table"></div>',
)
csvpath = ROOT / "map_results.csv"
with csvpath.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(best[0]))
    writer.writeheader()
    writer.writerows(best)
(ROOT / "map_results.json").write_text(
    json.dumps(best, ensure_ascii=False, indent=2), encoding="utf8"
)
css = """*{box-sizing:border-box}body{margin:0;background:#edf2f6;color:#182d3d;font-family:"Microsoft YaHei",sans-serif;line-height:1.75}header{background:#123d54;color:white;padding:26px max(5vw,20px)}main{max-width:1250px;margin:auto;padding:22px}section{background:white;border-radius:12px;padding:34px;margin:0 0 24px;box-shadow:0 2px 14px #18364a08}h1{font-size:27px;margin:0}h2{font-size:26px;line-height:1.4}h3{font-size:22px;margin:0 0 20px;border-left:5px solid #087fa7;padding-left:14px}p{margin:15px 0}img{width:100%;height:auto;display:block}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #dce5eb;text-align:left}th{background:#e9f1f6;color:#17445c}td{vertical-align:top}.scroll{overflow:auto}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.cards div{padding:17px;background:#f0f6fa;border-radius:8px}.cards b{display:block;font-size:30px;color:#087fa7}.callout{padding:18px;border-left:4px solid #d98a2d;background:#fff4e4}.routes,.formula{background:#edf6fa;padding:16px}.controls{display:flex;gap:18px;flex-wrap:wrap;position:sticky;top:0;background:white;padding:12px;z-index:2}select,input,button{font:inherit;padding:8px;border:1px solid #b9cbd5;border-radius:5px}input{max-width:100%}button{background:#0a6b8c;color:white;cursor:pointer;margin:10px}nav{display:flex;gap:15px;flex-wrap:wrap}nav a{color:#d8f1ff}a{color:#087fa7}.eyebrow{color:#547082}footer{text-align:center;padding:20px;color:#637886}[hidden]{display:none!important}@media(max-width:650px){main{padding:10px}section{padding:18px}.cards{grid-template-columns:repeat(2,1fr)}h2{font-size:22px}.controls{position:static;gap:6px}}@media print{body{background:white}section{box-shadow:none;break-inside:avoid}.controls,button,input{display:none}.map[hidden]{display:block!important}}"""
script = """const data=JSON.parse(document.getElementById('payload').textContent);function update(){const g=Number(document.getElementById('topo').value)*5+Number(document.getElementById('gdd').value),m=document.getElementById('metric').value;document.querySelectorAll('.map').forEach(x=>x.hidden=x.dataset.map!==g+'_'+m)}['topo','gdd','metric'].forEach(x=>document.getElementById(x).onchange=update);update();function rows(){const q=document.getElementById('search').value.toLowerCase();const d=data.filter(r=>(r.id+' '+r.topology+' '+(r.GDD_ps2>=0?'+':'')+r.GDD_ps2.toFixed(1)).toLowerCase().includes(q));document.getElementById('data-table').innerHTML='<div class="scroll"><table><tr>'+['编号','顺序','OC','GDD/ps²','pump/mW','平均能量/nJ','末圈峰数','CV','状态'].map(x=>'<th>'+x+'</th>').join('')+'</tr>'+d.map(r=>'<tr>'+[r.id,r.topology,Math.round(r.oc*100)+'%',r.GDD_ps2,r.pump_mW,r.numerical_pass?(r.energy_mean_pJ/1000).toFixed(4):'未解析',r.numerical_pass?r.peaks:'—',r.numerical_pass?r.energy_cv.toExponential(2):'—',r.screen_candidate?'初筛通过':r.numerical_pass?'未通过稳定目标组合':r.status].map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</table></div>'}document.getElementById('search').oninput=rows;rows();document.getElementById('download').onclick=()=>{const keys=Object.keys(data[0]);const text='\\uFEFF'+[keys.join(','),...data.map(r=>keys.map(k=>JSON.stringify(r[k])).join(','))].join('\\r\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'}));a.download='map_results.csv';a.click();URL.revokeObjectURL(a.href)};"""
body = "".join(
    f'<section id="s{i+1}"><h3>{i+1:02d}　{title}</h3>{content}</section>'
    for i, (title, content) in enumerate(sections)
)
page = (
    '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>20.42m 双拓扑色散能量地图</title><style>'
    + css
    + '</style><header><h1>原生10 MHz｜0.1–0.5 nJ 双拓扑搜索</h1><nav><a href="#s1">结果</a><a href="#s2">腔长</a><a href="#s6">地图</a><a href="#s'
    + str(len(sections))
    + '">数据</a></nav></header><main>'
    + body
    + '</main><footer>可离线分发 · 图像与结果数据全部内嵌 · 仿真结果不等同实物认证</footer><script type="application/json" id="payload">'
    + json.dumps(best, ensure_ascii=False)
    + "</script><script>"
    + script
    + "</script></html>"
)
out = ROOT / "原生10MHz_双拓扑色散能量地图_0.1至0.5nJ.html"
out.write_text(page, encoding="utf8")
print(
    out,
    "sections",
    len(sections),
    "valid",
    len(valid),
    "target single",
    len(single),
    "candidates",
    len(candidates),
    flush=True,
)
