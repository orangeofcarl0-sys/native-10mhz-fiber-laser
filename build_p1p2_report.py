"""Create a self-contained report fragment from actual validation records."""

import base64
import io
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import ROOT


def embedded(fig):
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return (
        '<img alt="算法验证图" src="data:image/png;base64,'
        + base64.b64encode(stream.getvalue()).decode()
        + '">'
    )


sat = json.loads((ROOT / "satellite_validation.json").read_text())
gpu = json.loads((ROOT / "resident_validation.json").read_text())
fig, ax = plt.subplots(figsize=(10, 4.3))
for row in sat[1:]:
    ax.plot(
        [x["round"] for x in row["trace"]],
        [x["ratio"] / row["initial_ratio"] for x in row["trace"]],
        label=f"initial satellite/main = {row['initial_ratio']:g}",
    )
ax.set(
    xlabel="Additional round trips",
    ylabel="(satellite/main) / initial ratio",
    title="Shared EDF gain: transient remote-satellite response",
)
ax.grid(alpha=0.25)
ax.legend()
ratio_image = embedded(fig)
fig, ax = plt.subplots(figsize=(10, 4.3))
for row in (sat[0], sat[-1]):
    ax.plot(
        [x["round"] for x in row["trace"]],
        [x["primary_pJ"] for x in row["trace"]],
        label=f"initial satellite/main = {row['initial_ratio']:g}",
    )
ax.set(
    xlabel="Additional round trips",
    ylabel="Main output pulse energy (pJ)",
    title="Primary output energy during remote-satellite perturbation",
)
ax.grid(alpha=0.25)
ax.legend()
energy_image = embedded(fig)
rows = "".join(
    "<tr><td>"
    + r["topology"]
    + "</td><td>"
    + f"{r['seconds_median']['host']:.3f}"
    + "</td><td>"
    + f"{r['seconds_median']['resident']:.3f}"
    + "</td></tr>"
    for r in gpu
)
html = (
    """<p>本次针对 P1 稳定性诊断与 P2 计算路径进行改造。物理参数、20.42 m 总腔长与两种拓扑保持原定义；下图是新增实算结果。</p>
<h4>远端卫星怎样与主脉冲竞争增益</h4>
<p>两个互不重叠的时间窗口共用一条 EDF 纵向反转分布。每个 EDF 单元先分别传播两个窗口，再把各窗口的频谱加权平均功率相加，仅更新一次反转和泵浦。窗口之间不做场相加；近邻脉冲则在同一窗口中相干叠加。</p>
<p class="formula">P<sub>mid</sub> = Σ<sub>窗口 m</sub> √(P<sub>in,m</sub>P<sub>out,m</sub>)。能量单位 pJ，乘以重复频率并换算为 W 后进入速率方程。弱卫星和主脉冲消耗同一个增益库。</p>
<p>沿用逐圈平均功率增益近似；不解析同一圈内脉冲先后造成的瞬时粒子数变化。假定远端窗口间隔远大于 CNT 恢复时间；未加入 ASE。该测试补充局部时间窗，仍不等同于完整 100 ns 全腔仿真。</p>
<h4>已存候选的 32 圈卫星扰动响应</h4>
<p>OC→CNT、OC30%、净色散 +0.2 ps²、pump20 mW；从第600圈归档场出发，dt=0.125 ps、8192点。卫星取主脉冲的缩放副本，初始能量比分别为 10⁻⁶、10⁻⁴、10⁻²，并设置零卫星对照。不重新归一化主脉冲。</p>
"""
    + ratio_image
    + energy_image
    + """
<p class="callout">观测：32圈后卫星/主脉冲之比降至初始值约85.7–85.8%。主脉冲能量同时由约143 pJ增加至171 pJ，卫星绝对能量也略有增长；比例下降并不等于卫星绝对衰减。主轨迹本身仍在演化，因此不能由此宣称卫星长期衰减或单脉冲稳定。还需要收敛基态、其他扰动形状/偏振与更长直接验证。</p>
<h4>复场、偏振与 Floquet 分析的使用条件</h4>
<p>原有复场残差仅移除共同时间平移与整体相位，继续检查完整纵向反转；新增归一化积分 Stokes 均值及跨度。近邻卫星工具支持指定延迟、能量比和相位，须另行检查周期窗回绕。</p>
<p>新增无矩阵中心差分 Arnoldi 工具，并以已知线性映射检验特征值。输入必须是无量纲实状态、去相位/时间规范后的固定网格一周期返回映射；先验证基态残差，再投影用户提供的中性方向。当前没有通过收敛认证的激光基态，因此未计算或报告激光腔 Floquet 乘子。</p>
<h4>GPU 常驻与 EDF 频域复用的验证</h4>
<p>EDF 单元复用第一半步输入频谱和最后半步输出频谱，FFT/IFFT由6次降为4次；独立标量参考保留原实现。新增 GPU 常驻接口，仅分块下载紧凑诊断、显式保存检查点；CNT 非法透射率标记为非有限值，GPU 边界/有限性失败永久锁定并保留上一个合格圈状态。</p>
<p>CPU批量对独立标量参考、两拓扑GPU对CPU均通过；GPU最大相对场误差约2.33×10⁻¹²。18项单元测试通过。以下是8案例×2048点×10圈、3次计时中位数；常驻计时包含边界诊断和最终检查点。</p>
<table><tr><th>拓扑</th><th>主机接口 / s</th><th>常驻接口 / s</th></tr>"""
    + rows
    + """</table>
<p>当前小批量未显示一致显著加速，不能据此声称整体扫描提速。常驻路径用于固定网格初筛，仅在圈末检测边界；CPU自适应回放仍用于严格复核。本节记录上一轮基线；后续已实现CNT prefix与频谱传递，详见第30节。FP32和自动多保真调度仍未实施。</p>"""
)
(ROOT / "p1p2_algorithm_validation.html").write_text(html, encoding="utf8")
print("Wrote p1p2_algorithm_validation.html")
