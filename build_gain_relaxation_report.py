"""Standalone gain-age validation report; never overwrites the full scan."""

import json, base64, csv
import numpy as np
from scipy.signal import find_peaks
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import ROOT


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert (
        manifest["status"] != "running"
    ), "Wait for completed or explicitly bounded run"
    terminal_check = ROOT / "terminal_grid_probe.json"
    if terminal_check.exists():
        manifest["final_grid_probe"] = json.loads(terminal_check.read_text())
    trace = json.loads((ROOT / "trace.json").read_text())
    with np.load(ROOT / "checkpoint.npz") as saved:
        age = saved["age"].copy()
        pop = saved["pop"].copy()
        a = saved["a"].copy()
        dt = float(saved["dt"])
    assert len(trace) == manifest["accepted_additional_rounds"]
    assert np.all(np.diff([r["gain_age_min"] for r in trace]) > 0)
    assert np.isclose(np.exp(-age.min()), manifest["gain_memory_max"])
    with (ROOT / "trace.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    periods = manifest["final_periods"]
    tail = np.array([r["energy_pJ"] for r in trace[-500:]]) / 1000
    slow_energy = np.array([r["energy_pJ"] for r in trace[-2000:]])
    slow_energy -= slow_energy.mean()
    autocorrelation = np.fft.irfft(
        abs(np.fft.rfft(slow_energy, n=2 * len(slow_energy))) ** 2
    )[: min(501, len(slow_energy))]
    autocorrelation /= max(autocorrelation[0], 1e-250)
    lags = find_peaks(autocorrelation[17:])[0] + 17
    slow_peaks = sorted(
        [
            dict(lag_rounds=int(k), energy_correlation=float(autocorrelation[k]))
            for k in lags
        ],
        key=lambda r: -r["energy_correlation"],
    )[:3]
    result = dict(
        total_rounds=manifest["total_rounds"],
        additional_rounds=len(trace),
        elapsed_physical_ms=len(trace) * (1.4682 * 20.42 / 299792458) * 1000,
        gain_age_min=manifest["gain_age_min"],
        gain_memory_max=manifest["gain_memory_max"],
        tail500_energy_min_nJ=float(tail.min()),
        tail500_energy_max_nJ=float(tail.max()),
        tail500_energy_cv=float(tail.std() / tail.mean()),
        tail500_peaks_min=min(r["peaks"] for r in trace[-500:]),
        tail500_peaks_max=max(r["peaks"] for r in trace[-500:]),
        minimum_field_residual_over_periods=min(
            (r["field"] for r in periods), default=None
        ),
        gain_age_cells=age.ravel().tolist(),
        inversion_cells=pop.ravel().tolist(),
        events=manifest["events"],
        status=manifest["status"],
        classification=manifest["classification"],
        certified_stable=False,
    )
    result["slow_energy_correlation_candidates"] = slow_peaks
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 12,
        }
    )

    def figure(name, caption):
        plt.tight_layout()
        plt.savefig(ROOT / name, dpi=150)
        plt.close()
        data = base64.b64encode((ROOT / name).read_bytes()).decode()
        return f'<figure><img alt="{caption}" src="data:image/png;base64,{data}"><figcaption>{caption}</figcaption></figure>'

    x = np.array([r["round"] for r in trace])
    body = f"""<h1>g08_c53：累计增益年龄与长程验证</h1>
<p>固定CNT→OC、OC80%、净GDD+0.2 ps²、pump50 mW。由本轮完整地图第600圈的复场、逐段EDF反转和CNT状态直接续算；不使用增益加速，不改变器件参数，不开展参数支路追踪。</p>
<aside>本次接受{len(trace):,}个新增真实往返，总圈数{manifest['total_rounds']:,}。累计增益年龄最小值为{age.min():.4f}，条件松弛系数上界为{np.exp(-age.min()):.4g}。运行状态：{manifest['status']}。当前不认证稳定单脉冲。</aside>
<h2>1. 累计增益年龄的定义和适用范围</h2>
<p>逐EDF单元累积 Γⱼ=ΣₖBⱼ,ₖTᴿ，再取Γ_min=minⱼΓⱼ；M_max=exp(−Γ_min)。先按单元求和再取最小值，不能用每圈最小B的和替代。这里的M是沿已实现A、B历史的齐次反转松弛系数；完整耦合敏感度还包含光场与反转对A、B的反馈，不能把M叫作整个激光器已忘记初态的百分比。</p>
<p>前600圈未保存逐段B历史，因此Γ从续算开始的零计起，不用末态τ倒推过去。只累计接受的物理圈，触边失败的试探圈、冻结搜索与加速迭代均不计。当前时间网格变化不改变EDF单元身份，因此保留年龄；自适应入口若改变认证网格则保守重置。</p>
"""
    plt.figure(figsize=(10, 5))
    plt.plot(x, [r["gain_age_min"] for r in trace])
    plt.axhline(7, color="tab:red", ls="--", label="条件年龄门槛 Γ=7")
    plt.xlabel("总往返圈数（含已有600圈）")
    plt.ylabel("从第600圈起累计的 Γ_min")
    plt.legend()
    plt.grid(alpha=0.2)
    body += figure(
        "gain_age.png",
        "Γ=7是条件松弛门槛，不是稳定性证明；图中年龄不包含未知的前600圈历史。",
    )
    plt.figure(figsize=(10, 5))
    plt.plot(x, [r["gain_memory_max"] for r in trace])
    plt.yscale("log")
    plt.axhline(np.exp(-7), color="tab:red", ls="--")
    plt.xlabel("总往返圈数")
    plt.ylabel("条件松弛系数 exp(−Γ_min)")
    plt.grid(alpha=0.2)
    body += figure(
        "gain_memory.png",
        "该曲线不包含耦合反馈的切线增长，不能直接用作完整初态扰动的衰减率。",
    )
    body += f"""<h2>2. 输出能量和局部峰结构的演化</h2><p>末500圈能量范围{tail.min():.4g}–{tail.max():.4g} nJ，CV={tail.std()/tail.mean()*100:.2f}%；输出局部峰数范围{result['tail500_peaks_min']}–{result['tail500_peaks_max']}。峰数沿用高度和突出度各10%的局部峰判据，不把局部峰数直接当作分离孤子数。</p>"""
    plt.figure(figsize=(10, 5))
    plt.plot(x, np.array([r["energy_pJ"] for r in trace]) / 1000, lw=0.8)
    plt.axhspan(0.1, 0.5, color="teal", alpha=0.15, label="目标0.1–0.5 nJ")
    plt.xlabel("总往返圈数")
    plt.ylabel("输出能量 / nJ")
    plt.legend()
    plt.grid(alpha=0.2)
    body += figure(
        "energy.png", "真实时间能量轨迹；没有归一化输出能量或动态匹配腔损耗。"
    )
    plt.figure(figsize=(10, 5))
    plt.plot(x, [r["peaks"] for r in trace], lw=0.7)
    plt.xlabel("总往返圈数")
    plt.ylabel("输出强度局部峰数")
    plt.grid(alpha=0.2)
    body += figure("peaks.png", "局部峰结构随演化变化；该窗口不是整个约100 ns腔周期。")
    plt.figure(figsize=(10, 5))
    plt.plot(x, [r["inversion_gap"] for r in trace])
    plt.xlabel("总往返圈数")
    plt.ylabel("瞬时反转平衡偏差 max|N−A/B|")
    plt.grid(alpha=0.2)
    body += figure(
        "inversion_gap.png",
        "瞬时平衡偏差可排除完整period-1固定点，但不能单独否定周期呼吸态。",
    )
    body += """<h2>3. 周期1–16的复场与反转重复检查</h2><p>在同一腔内参考截面检查完整两偏振复场、反转剖面和CNT状态。仅移除共同时移与全局相位，不消去偏振旋转。每500圈检查最近65圈，末态再检查一次；至少覆盖周期16的四个周期。没有命中1–16圈递归不能推导无更长周期或混沌。</p><p>采用原有严格筛选量级：复场/强度/Stokes≤1e−3、能量逐周期变化≤1e−4、反转相对误差≤1e−4且最大绝对差≤1e−5、CNT状态差≤1e−8。另做局部双网格误差检查，不因误差大而放宽门槛。它们是明确的筛选要求，尚不是通用工程容差或完整稳定认证。</p><div class="scroll"><table><tr><th>周期p</th><th>复场r_A</th><th>强度r_I</th><th>Stokes</th><th>反转相对差</th><th>能量重复误差</th><th>筛选</th></tr>"""
    for r in periods:
        body += f'<tr><td>{r["period"]}</td><td>{r["field"]:.3g}</td><td>{r["intensity"]:.3g}</td><td>{r["stokes"]:.3g}</td><td>{r["population"]:.3g}</td><td>{r["energy"]:.3g}</td><td>{"通过候选门槛" if r["passes_screen"] else "未通过"}</td></tr>'
    body += "</table></div><p>即使通过，也只称有限记录递归候选；还需持续保持、独立扰动和全腔卫星竞争检查，才讨论吸引稳定性。</p>"
    body += (
        "<p>为避免把1–16圈之外的慢振荡漏掉，另列末2000圈输出能量的自相关峰。该诊断只产生慢周期线索，不能替代该延迟下完整复场/反转的重复检查："
        + json.dumps(slow_peaks, ensure_ascii=False)
        + "</p>"
    )
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(autocorrelation)), autocorrelation)
    plt.xlabel("延迟 / 往返圈数")
    plt.ylabel("末2000圈输出能量的归一化自相关")
    plt.grid(alpha=0.2)
    body += figure(
        "slow_energy_correlation.png",
        "自相关峰表示能量层面的重复线索，不是稳定周期吸引子的认证。",
    )
    body += "<h2>4. 数值回退、网格误差与结论限制</h2><p>同时检查腔内返回场和输出场的时间/谱边界；触边后退回未触边输入，再细化dt或扩大窗口。EDF纵向步长和无源步长仍是固定网格；这次没有声称完成每个子步的自适应误差控制。</p><ul>"
    for event in manifest["events"]:
        body += f'<li>总第{event["at_total_round"]}圈之后重试：{event["action"]}，dt={event["dt_ps"]} ps，N={event["n"]}。</li>'
    body += (
        "</ul><p>初始及末态用8圈dt减半、EDF及无源步长同时减半作局部对照。这仅检验局部差异，不能证明整个长期轨迹全局收敛。</p><pre>"
        + json.dumps(
            {
                "initial": manifest["initial_grid_probe"],
                "final": manifest["final_grid_probe"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "</pre>"
    )
    if not manifest["final_grid_probe"].get("below_screen_tolerances", False):
        body += "<aside>末态网格复核未通过或受资源限制：晚期行为只作为当前离散模型轨迹报告，不据此确认物理吸引态、呼吸机制或混沌。</aside>"
    body += """<h2>5. 后续动作与未执行的试验</h2><p>本轮没有将未收敛末态用于50→30 mW的支路追踪，也没有改变CNT饱和参数。若无可信稳定锚点，应先处理数值收敛或延长时间，而不是沿移动暂态构造稳态分岔图。</p><p>关于附件的平均损耗匹配：Psat=80 W时新透射可能比基准更低，T_ref/T_new会大于1，不能叫作被动衰减器。要全组被动匹配，应选T_target=minᵢTᵢ，在所有组加固定Lᵢ=T_target/Tᵢ≤1。匹配只在参考场成立，不保证后续不同状态的平均损耗仍相同；该试验尚未执行。</p><p>原始检查点保留A_x、A_y、N(z)、q及逐单元Γ，曲线有CSV导出。原640点地图与诊断报告保持原样。</p>"""
    data = base64.b64encode((ROOT / "trace.csv").read_bytes()).decode()
    body += f'<p><a download="gain_relaxation_trace.csv" href="data:text/csv;base64,{data}">下载全部接受圈的轨迹 CSV</a></p>'
    css = 'body{font:17px/1.8 "Microsoft YaHei",sans-serif;background:#eef3f6;color:#20384c;margin:0}main{max-width:1120px;margin:auto;background:white;padding:32px}h1{font-size:28px}h2{font-size:23px;margin-top:35px;border-left:5px solid #0088a5;padding-left:14px}aside{background:#fff2dc;padding:18px}img{width:100%;height:auto}figure{margin:25px 0}figcaption{font-size:14px;color:#567}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:8px;border:1px solid #d9e2e8}th{background:#eaf3f6}.scroll,pre{overflow:auto}pre{font-size:13px}a{color:#087a9c}@media(max-width:600px){main{padding:15px}}'
    (ROOT / "g08_c53_增益年龄与长程验证.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>g08_c53增益年龄与长程验证</title><style>'
        + css
        + "</style><main>"
        + body
        + "</main></html>",
        encoding="utf8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
