"""Self-contained HTML fragment from measured GPU ablations and regression checks."""

import base64, io, json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from config import ROOT


def image(fig, alt):
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return (
        '<img alt="'
        + alt
        + '" src="data:image/png;base64,'
        + base64.b64encode(stream.getvalue()).decode()
        + '">'
    )


data = json.loads((ROOT / "spectral_gpu_benchmark.json").read_text())
checks = json.loads((ROOT / "spectral_gpu_validation.json").read_text())
workflow = json.loads((ROOT / "resident_workflow_validation.json").read_text())
rows = data["records"]
modes = ["reference", "spectral_serial_cnt", "spectral_prefix_cnt", "spectral_fused"]
labels = [
    "Reference (801fd03)",
    "Spectral carry",
    "+ CNT prefix",
    "+ EDF fusion / in-place Kerr",
]
colors = ["#9caebd", "#5d9bc4", "#30a6a0", "#174b6b"]
fig, ax = plt.subplots(figsize=(11, 5))
positions = np.arange(4)
for j, (mode, label, color) in enumerate(zip(modes, labels, colors)):
    values = [r["timings"][mode]["cuda_ms_median"] for r in rows]
    ax.bar(positions + (j - 1.5) * 0.2, values, width=0.18, label=label, color=color)
ax.set_xticks(positions, [f"{r['topology']}\nN={r['n']}" for r in rows])
ax.set(
    ylabel="CUDA event time (ms / round trip)",
    title="Matched FP64 propagation: 64 cavities",
)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
ax.grid(axis="y", alpha=0.2)
ablation = image(fig, "四种GPU实现的同输入整圈耗时")
fig, ax = plt.subplots(figsize=(10, 4.5))
for j, (mode, label, color) in enumerate(
    zip(
        ["serial", "prefix"],
        ["Serial CNT", "Fused block-prefix CNT"],
        ["#9caebd", "#174b6b"],
    )
):
    ax.bar(
        np.arange(4) + (j - 0.5) * 0.3,
        [r["cnt"][mode]["cuda_ms_median"] for r in rows],
        width=0.28,
        label=label,
        color=color,
    )
ax.set_xticks(np.arange(4), [f"{r['topology']}\nN={r['n']}" for r in rows])
ax.set(
    ylabel="CUDA event time (ms)",
    title="CNT-only microbenchmark: identical fields and initial absorption",
)
ax.legend()
ax.grid(axis="y", alpha=0.2)
cnt_image = image(fig, "串行CNT与融合前缀CNT独立计时")
table = ""
for r in rows:
    values = [r["timings"][m]["cuda_ms_median"] for m in modes]
    table += (
        "<tr><td>"
        + r["topology"]
        + " / "
        + str(r["n"])
        + "</td>"
        + "".join(f"<td>{v:.2f}</td>" for v in values)
        + f"<td>{values[0]/values[-1]:.2f}×</td></tr>"
    )
endtable = ""
for r in workflow["rows"]:
    old, new = r["seconds_median"]["reference"], r["seconds_median"]["spectral"]
    endtable += f"<tr><td>{r['samples']}</td><td>{old:.3f}</td><td>{new:.3f}</td><td>{old/new:.2f}×</td><td>{r['field_relative_error']:.2e}</td></tr>"
max_one = max(r["errors"]["spectral_fused"] for r in rows)
html = f"""<p>本节分析附件提出的频域传递、CNT并行扫描与EDF融合，并给出当前代码实测。硬件：{data['device']}；CuPy {data['cupy']}。所有结果使用FP64，未启用fast math。</p>
<h4>当前基线与变换次数的对应关系</h4>
<p>附件的182次/圈针对更早版本。上一轮已把EDF从6次/单元减至4次，本轮开始时实际为：无源段92次 + EDF 60次 = <b>152次/圈</b>。五个现有GDD切片均有56个Kerr步。</p>
<table><tr><th>接口</th><th>FFT/IFFT次数</th><th>额外转换来源</th></tr><tr><td>旧时域核心</td><td>152</td><td>光纤段边界及EDF单元边界往返</td></tr><tr><td>新时域兼容接口</td><td>116</td><td>112次Kerr + 2次CNT + 输入FFT/输出IFFT</td></tr><tr><td>纯频谱返回映射</td><td>114</td><td>只有Kerr和CNT需要时域</td></tr><tr><td>跨圈常驻并做圈末边界检查</td><td>115</td><td>额外1次IFFT检查时域边缘；频谱诊断直接复用</td></tr></table>
<h4>线性传播与非线性操作的实现次序</h4>
<p class="formula">频谱 → 线性半步 → IFFT → Kerr → FFT → 线性半步 → 下一段频谱。只有PC/OC/CNT位置统一转入时域，分光顺序仍由OC→CNT或CNT→OC决定。</p>
<p>相邻线性操作之间直接传递频谱，去掉无实际操作的IFFT→FFT，未更改Strang分裂、步长或物理参数。保留原模型的偏振约定：无源段Kerr在段内基底计算，EDF Kerr在实验室基底计算；不交换Jones矩阵的次序。</p>
<p>EDF的输出加权功率直接作为下一单元输入。融合内核分别做加权归约、增益半步及反转/泵浦更新；同单元两个半步均使用更新前反转。无源整步系数预计算，Kerr采用二维网格原地写回。</p>
<h4>逐项改造后的整圈耗时</h4>
<p>每组64案例、相同输入、相同精度，预热2次，交错随机顺序测7次，表内为CUDA Event中位数。计时包含设备内输入副本，不包含编译和主机数据传输；每次样本均保存在JSON。未找到可调用的Nsight CLI，这不是Nsight的逐内核占比统计。</p>
{ablation}<div class="scroll"><table><tr><th>拓扑 / 点数</th><th>旧核心/ms</th><th>频谱传递/ms</th><th>再加CNT前缀/ms</th><th>再加EDF融合与原地Kerr/ms</th><th>整圈提速</th></tr>{table}</table></div>
<p>这是真实整圈消融，不是由152/116推算的速度。频谱传递与EDF融合是主要收益；性能受批量、GPU时钟及网格影响，不能直接外推到其他设备或全部搜索流程。</p>
<h4>CNT前缀扫描的并行方式与局部收益</h4>
<p>原实现每腔一个线程串行遍历所有采样。新实现每腔一个256线程块：各线程先汇总连续片段的仿射递推，再作有序块内前缀扫描，得到每片段入口吸收量后展开。融合计算双偏振功率、透射场、最终吸收量及能量/峰值等诊断，不保存完整power和transmission临时数组。</p>
<p class="formula">q′ = a q + b；(a₂,b₂)∘(a₁,b₁) = (a₂a₁, a₂b₁+b₂)。复合顺序不交换，不除以极小累乘；空片段使用恒等映射。</p>
{cnt_image}<p>CNT单独计时约提速12–15倍，但原CNT只占本次整圈耗时的一小部分，因此不等于整圈提速12倍。微基准与整圈中调度/缓存环境不同，不能将各部件计时直接相加。</p>
<h4>包含上传、诊断与检查点的常驻工作流</h4>
<p>下表为64案例、OC→CNT、连续10圈，包含构造/上传、每圈相同紧凑诊断、最终检查点下载，3次交替顺序计时中位数。与旧参考核心比较，而非只测一个CUDA内核。</p>
<table><tr><th>点数</th><th>旧路径/s</th><th>新路径/s</th><th>提速</th><th>终态相对场误差</th></tr>{endtable}</table>
<h4>数值一致性和异常处理验证</h4>
<p>20项CPU测试通过。新增GPU测试覆盖36组CNT输入（17/257/2048/8192点，弱场、强场、不同dt）、两拓扑与两网格各20圈、非零双折射/DGD/偏振旋转、活动批量缩小、冻结增益和非法透射率回退。</p>
<table><tr><th>对照</th><th>最大相对场误差</th><th>适用范围</th></tr><tr><td>默认参数单圈新旧GPU核心</td><td>{max_one:.2e}</td><td>两拓扑、两网格</td></tr><tr><td>CNT前缀对CPU递推</td><td>{checks['max_cnt_field_relative']:.2e}</td><td>36组测试</td></tr><tr><td>加入偏振扰动的20圈轨迹</td><td>{checks['max_20round_field_relative']:.2e}</td><td>4组批量轨迹，每组20圈；非长期吸引态认证</td></tr></table>
<p class="callout">浮点运算次序改变后不要求逐位一致。短程误差很小，不代表分岔附近上千圈轨迹必然重合；物理结论仍需网格收敛、慢增益时间尺度与扰动验证。</p>
<h4>扫描入口与认证边界</h4>
<p>原MAP_GPU=1入口默认使用新核心，原精确CPU诊断保持不变。新增scan_resident.scan入口固定64槽、每20圈下载一次紧凑诊断，最后下载场并计算SciPy峰数；每案例记录合格圈数与失败标记。固定容量避免FFT形状反复变化，但仍计算失效槽。</p>
<p>已完成一个64案例组的40圈工作流检查：{workflow['screen_smoke']['failed_cases']}例触发圈末边界/有限性失败，剩余状态保存供进一步检查。粗略局部峰数不含prominence/平台峰规则，末圈精确峰数也不等于长期稳定；本次未重跑640点长期地图。</p>
<p>CUDA Graph、cuFFT callbacks、显式plan、动态bucket压缩和FP32仍未纳入本次实现。未将FP32初筛失败作为物理不可行结论。计时方法参照<a href="https://docs.cupy.dev/en/stable/user_guide/performance.html">CuPy官方性能说明</a>。</p>"""
(ROOT / "gpu_optimization_validation.html").write_text(html, encoding="utf8")
print("Wrote gpu_optimization_validation.html")
