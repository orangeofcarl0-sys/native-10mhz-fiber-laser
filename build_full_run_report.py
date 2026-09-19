"""Report the complete new scan, with actual fine-grid output diagnostics."""

import base64, io, json
from datetime import datetime
import numpy as np
from scipy.io import loadmat
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import ROOT


def picture(fig, label):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return (
        '<img alt="'
        + label
        + '" src="data:image/png;base64,'
        + base64.b64encode(buffer.getvalue()).decode()
        + '">'
    )


summary = json.loads((ROOT / "full_run_summary.json").read_text(encoding="utf8"))
manifest = json.loads((ROOT / "run_manifest.json").read_text(encoding="utf8"))
plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 12,
    }
)
elapsed = (
    datetime.fromisoformat(manifest["finished_utc"])
    - datetime.fromisoformat(manifest["started_utc"])
).total_seconds()
table = ""
comparison = ""
for key, label in [
    ("coarse", "粗网格：Δt=0.5 ps，2048点"),
    ("fine", "细网格：Δt=0.125 ps，8192点"),
]:
    row = summary["grids"][key]
    now = row["current"]
    before = row["previous"]
    table += f"<tr><td>{label}</td><td>{now['cases']}</td><td>{now['numerical_pass']}</td><td>{now['target_single']}</td><td>{now['short_screen']}</td><td>{now['actual_case_rounds']:,}</td></tr>"
    comparison += f"<tr><td>{label}</td><td>{before['numerical_pass']} → {now['numerical_pass']}</td><td>{before['target_single']} → {now['target_single']}</td><td>{row['same_stop_status']}/640</td><td>{row['first_round_energy_max_relative']:.2e}</td><td>{row['first20_energy_max_relative']:.2e}</td></tr>"
html = f"""<p>本次使用新FP64 GPU核心，完整重跑2种拓扑 × 8档OC × 5档净色散 × 8档泵浦，共640个物理组合；粗、细两套网格合计1280条轨迹、20组结果。每点最多600圈，初态和停止门槛与上一版保持一致。</p>
<p>开始：{manifest['started_utc']}；结束：{manifest['finished_utc']}。扫描与保存耗时约{elapsed/60:.1f}分钟。每组均保存JSON摘要、MAT逐圈诊断及末态；文件校验和、代码指纹与完成清单单独归档。</p>
<h4>完整重算覆盖与筛选结果</h4>
<table><tr><th>网格</th><th>参数点</th><th>通过短时边界检查</th><th>目标能量且连续单峰</th><th>全部短时初筛门槛通过</th><th>实际累计案例圈数</th></tr>{table}</table>
<p>“通过短时边界检查”要求完成600圈且末段未触边；目标单峰还要求末100圈能量均在0.1–0.5 nJ、末64圈为单峰。全部短时门槛再加能量CV&lt;10⁻⁴、最大强度残差&lt;10⁻³。该旧地图初筛判据本身也不等于P1完整复场/慢增益/远端卫星认证。</p>
<p class="callout">完成全部参数点的计算流程，不等于每个点都获得物理解。触发频谱或时间边界的点仍标为数值未解析；没有把它们补成稳定输出，也没有把没有候选解释成架构不可行。</p>
<h4>相同初态下的新旧结果对照</h4>
<p>逐组核对了20组初始复场、EDF反转和全部参数标签，完全一致。以下比较旧版与本次实算，不混用旧轨迹填充新结果。</p>
<div class="scroll"><table><tr><th>网格</th><th>边界通过数：旧→新</th><th>目标单峰数：旧→新</th><th>停止状态一致数</th><th>首圈能量最大相对差</th><th>前20圈能量最大相对差</th></tr>{comparison}</table></div>
<p>初始短程能量一致性与最终分类分别检查。局部浮点运算次序改变后，长程非线性轨迹可能放大微小差异；不能用单个终态能量偏差直接推断算法改变了物理模型。</p>"""
differences = summary["changed_cases"]
if differences:
    html += "<h4>停止状态或地图分类改变的参数点</h4><p>这些差异保留在新地图中，不用旧结果替换。图示新旧共同计算区间内的能量相对差，用于区分首圈实现差异与后续差异增长；差异增长本身不构成混沌或具体分岔机制的证明。</p>"
    html += "<table><tr><th>网格/编号</th><th>旧状态/圈数</th><th>新状态/圈数</th><th>前20圈最大相对差</th><th>首次超过1%的圈数</th></tr>"
    labels = {
        "completed": "完成",
        "spectral_boundary": "频谱边界",
        "time_boundary": "时间边界",
        "decayed_or_nonfinite": "衰减/非有限",
    }
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for row in differences:
        html += f"<tr><td>{row['grid']}/{row['id']}</td><td>{labels[row['old_status']]}/{row['old_rounds']}</td><td>{labels[row['new_status']]}/{row['new_rounds']}</td><td>{row['first20_energy_max_relative']:.2e}</td><td>{row['first_difference_above_1percent'] if row['first_difference_above_1percent'] is not None else '共同区间未超过'}</td></tr>"
        curve = np.array(row["energy_relative_difference"])
        ax.semilogy(
            np.arange(1, len(curve) + 1),
            np.maximum(curve, 1e-18),
            label=row["grid"] + "/" + row["id"],
        )
    ax.axhline(0.01, color="#a35b30", ls="--", label="能量相对差1%")
    ax.set(
        xlabel="传播圈数",
        ylabel="新旧输出能量的相对差",
        title="分类改变点的新旧轨迹差异增长",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    html += "</table>" + picture(fig, "分类改变点的逐圈差异")
points = summary["grids"]["fine"]["target_single_points"]
if points:
    best = points[0]
    group = best["group"]
    index = best["index"]
    m = loadmat(
        ROOT / f"fine_group_{group:02d}.mat",
        simplify_cells=True,
        variable_names=["out", "trace", "completed", "t", "dt", "rep"],
    )
    count = int(m["completed"][index])
    trace = m["trace"][:count, index]
    label = f"{best['topology']}，OC={best['oc']:.0%}，GDD={best['GDD_ps2']:+.1f} ps²，pump={best['pump_mW']:g} mW"
    html += f"""<h4>本次细网格中目标单峰的代表输出</h4><p>按能量CV由小到大选择{best['id']}：{label}。末100圈能量{best['energy_min_pJ']/1000:.4f}–{best['energy_max_pJ']/1000:.4f} nJ，CV={best['energy_cv']:.2%}，末圈FWHM={best['FWHM_ps']:.3f} ps。它是目标附近的瞬态，不是已确定的稳定工作点。</p>"""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(np.arange(1, count + 1) / float(m["rep"]) * 1e6, trace[:, 0] / 1000, lw=1.7)
    ax.axhspan(0.1, 0.5, color="#54b3a0", alpha=0.15, label="目标能量范围")
    ax.set(
        xlabel="从初态起的演化时间 / μs",
        ylabel="OC输出单脉冲能量 / nJ",
        title=label + "：完整能量演化",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    html += picture(fig, "本次代表状态600圈能量演化")
    field = m["out"][index]
    power = np.sum(abs(field) ** 2, axis=0)
    t = m["t"]
    center = t[np.argmax(power)]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(t - center, power, lw=1.6)
    ax.set(
        xlabel="相对主峰时间 / ps",
        ylabel="瞬时输出功率 / W",
        title=f'{best["id"]}：第{count}圈输出脉冲（完整局部时间窗）',
    )
    ax.grid(alpha=0.2)
    html += picture(fig, "实际细网格末圈输出脉冲")
    spectrum = np.fft.fftshift(np.sum(abs(np.fft.fft(field, axis=-1)) ** 2, axis=0))
    freq = np.fft.fftshift(np.fft.fftfreq(len(t), float(m["dt"])))
    db = 10 * np.log10(np.maximum(spectrum / max(spectrum.max(), 1e-100), 1e-10))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(freq, db, lw=1.5)
    ax.set(
        xlabel="相对光学载频的频率偏移 / THz",
        ylabel="相对光谱功率 / dB",
        ylim=(-80, 3),
        title=f'{best["id"]}：与时域同一输出场的光谱包络',
    )
    ax.grid(alpha=0.2)
    html += picture(fig, "本次末圈输出场光谱包络")
    html += "<p>时域、光谱与能量轨迹来自同一条本次细网格记录。局部1.024 ns窗口的傅里叶包络不能解析10 MHz梳齿，也不能排除约100 ns全腔周期中的远端脉冲。</p>"
html += """<h4>结果管理与历史验证的范围</h4><p>当前报告主地图、表格和640点CSV已切换到本次细网格结果；另存1280行双网格CSV/JSON。旧报告和旧原始数据保留。第12–20节为明确标注的历史长程验证，不伪装成本次新续算。</p><p>本轮没有修改器件参数、CNT饱和能量、泵浦或耦合比例来增加候选数量；新的计算速度没有改变稳定性验收要求。</p>"""
html += (
    '<p><a download="all_1280_results.csv" href="data:text/csv;charset=utf-8;base64,'
    + base64.b64encode((ROOT / "all_1280_results.csv").read_bytes()).decode()
    + '">下载粗、细双网格1280行结果 CSV</a></p>'
)
(ROOT / "full_run_validation.html").write_text(html, encoding="utf8")
print("Wrote full_run_validation.html")
