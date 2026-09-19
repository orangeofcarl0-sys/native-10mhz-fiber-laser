"""Build a standalone interpretation report without changing the scan report."""

import base64, json
from config import ROOT

s = json.loads((ROOT / "mechanism_diagnostics.json").read_text(encoding="utf8"))
rows = {r["id"]: r for r in s["selected"]}


def figure(name, caption):
    data = base64.b64encode((ROOT / name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


table = ""
for name in ["g03_c11", "g08_c52", "g03_c58", "g02_c17", "g08_c53"]:
    r = rows[name]
    table += f'<tr><td>{name}</td><td>{r["energy_mean_pJ"]/1000:.3f}</td><td>{r["energy_cv"]*100:.2f}%</td><td>{r["CNT_peak_W"]:.2f}</td><td>{r["bleach_fraction_600"]*100:.2f}%</td><td>{r["inversion_gap"]:.3f}</td><td>{r["probe601_tau_max_us"]:.0f}</td></tr>'

body = (
    """<h1>原生10 MHz锁模搜索：结果成因与证据边界</h1>
<p>分析对象：2026-09-19完整细网格640点，以及已有长程验证。主问题是当前为什么没有稳定目标解，而不是如何放宽门槛让地图出现通过点。本次只做轨迹后处理和12个代表状态的一圈诊断，未执行新的长程支路追踪或CNT参数干预。</p>
<aside><b>当前可成立的结论：</b>既定参数、初态和时长下，没有找到经认证的0.1–0.5 nJ稳定单脉冲。目标单峰点的CNT调制较弱是事实；慢增益未平衡也是直接证据。现有数据还不足以认定存在高于0.5 nJ的最低稳定能量，或把失稳唯一归因于CNT。</aside>
<h2>1. 代表点的实际状态与统计口径</h2>
<p>能量和CV为第501–600圈统计；CNT峰值、实际漂白比例和反转偏差为第600圈。τ为从保存末态继续一圈（第601圈）计算的瞬时EDF有效恢复时间的空间最大值。它与第600圈统计分开标注，不把一圈诊断称为长程验证。不同列未必属于同一空间位置，不能将两个最大值当成同一EDF单元。</p>
<div class="scroll"><table><tr><th>编号</th><th>均值/nJ</th><th>能量CV</th><th>CNT峰值/W</th><th>可饱和损耗漂白比例</th><th>反转平衡偏差</th><th>τ最大值/μs，第601圈</th></tr>"""
    + table
    + """</table></div>
<p>细网格264点通过数值边界检查；8点末100圈能量全部位于目标带且末64圈单峰；严格短期筛选为0。全部8个目标单峰的CV为18.97%–43.99%，因此改为1%阈值仍无通过点。但这些状态均未完成增益平衡，不能把此表当成稳态相图。</p>
<h2>2. CNT调制强度与当前脉冲的匹配</h2>
<p>batch_engine.absorber及GPU版本实现 dq/dt=(q₀−q)/τ−qP/E_sat，强度透射率为T=1−q_ns−q；P为CNT入射处两偏振总功率，时间单位ps，能量单位pJ。参数为q₀=0.05、q_ns=0.10、τ=0.5 ps、E_sat=20 pJ，所以P_sat=E_sat/τ=40 W。</p>
<p>在CNT处功率变化慢于恢复时间时，q≈q₀/(1+P/P_sat)。输出FWHM不能直接替代CNT处脉宽，故本次同时读取原始trace中实际积分得到的q_min。g03_c11实际q_min=0.043676，与准稳态估计吻合：5%的可饱和损耗仅漂白12.65%，峰值透射率只增加0.632个百分点。第601圈按整脉冲能量加权的透射率提升约0.448个百分点；峰值提升不能当作整脉冲收益。</p>
<p>g08_c53的漂白比例约74.71%，确实处于更强调制区。但P/P_sat=1不是锁模阈值，弱于此值并不自动排除锁模。改变pump还同时改变反转、能量、非线性相移及增益滤波，不能用一个强饱和点的低CV证明CNT是唯一原因。</p>
"""
    + figure("cnt_history.png", "CNT实际损耗变量的演化；曲线来自保存的600圈轨迹。")
    + figure(
        "saturation_vs_cv.png",
        "264个边界通过点的诊断散点。横轴为末圈，纵轴为末100圈；属于相关性图，不是因果干预。",
    )
    + """
<h2>3. 慢增益收敛与600圈观察时长</h2>
<p>20.42 m腔的往返时间约100 ns，600圈约60 μs，末100圈只有10 μs。EDF按每个纵向单元更新归一化反转n：dn/dt=A−Bn，瞬时平衡n_eq=A/B，有效恢复时间τ_eff=1/B；A、B随局部泵浦及光谱加权信号变化。spectral_engine.fiber_spectrum保存max|n−n_eq|和maxτ_eff。</p>
<p>不能直接用10 ms上能级寿命断言需要10万圈，也不能忽略慢增益。根据当前状态补算，g03_c11的τ_eff,max约899 μs，g08_c52约627 μs，g08_c53约198 μs。60 μs分别只约为这些末态局部时间尺度的0.067、0.096、0.303倍；这是数量级比较，并非整个演化过程的累计收敛时间。</p>
<p>g08_c53末圈最大反转平衡偏差仍0.176。反转是0–1归一化占比，因此这是约17.6个百分点的局部平衡偏差，不是每圈漂移17.6%。其空间平均反转末段变化虽小，却可能掩盖不同单元的反向变化。应观察逐单元反转、全局增益与光场共同收敛。</p>
"""
    + figure(
        "gain_gap_history.png",
        "反转平衡偏差没有收敛到零；能量暂时平缓不能替代慢变量收敛。",
    )
    + figure(
        "energy_history.png", "g08_c53比两个目标点更平缓，但当前仍是有限时间轨迹。"
    )
    + """
<p>初始化map_initial以腔损耗估计统一纵向反转，并不是每个泵浦下的稳态反转分布；同一OC下不同pump共用此初始反转。这样有利于公平比较，却意味着扫描包含共同初态出发的储能释放和调整过程。600圈时的状态可能受初始储能影响。</p>
<h2>4. 低CV、高能量单峰与低CV多峰分别说明什么</h2>
<p>g08_c53的6个连续100圈窗口CV依次约75.19%、31.01%、16.14%、8.82%、2.25%、1.07%，支持它正在趋向更平缓状态。但末100圈端点能量仍从1.792升至1.872 nJ，增幅相对均值约4.36%。它值得优先续算；现在还不能称为稳定支路。</p>
<p>贴文所列g00_c49、g01_c50、g01_c41的多峰低CV数字基本吻合，但这三个点CNT峰值仅约10.15、3.46、3.59 W，实际漂白约16.11%、7.63%、7.98%。因此“异常/零色散区CNT已充分饱和”没有被这组数据证明。</p>
<p>程序peaks统计输出强度的局部极大值，阈值和突出度均为峰值的10%。8–10个峰不自动等于8–10个彼此分离的孤子；还可能是一个结构化脉冲包或时间分辨率下的尖峰。需要峰间距、逐峰能量、相位和传播中的分离行为，才能讨论孤子能量量子化。末段单峰也不能排除低于阈值的卫星。</p>
<h2>5. 数值边界、强度残差和历史续算的限制</h2>
<p>376/640个细网格点触发谱边界。实际统计为：10 mW是28/80，150 mW是78/80；OC20%是76/80，OC90%是19/80；GDD+0.2为49/128。贴文部分数字来自旧版本，趋势相同但应采用本次原始记录。</p>
<p>高泵浦、低输出耦合更容易触边，符合腔内功率与谱展宽的关联；也可能混合传播步长误差、离散混叠和模型带宽限制。当前边界检测每20圈在输出口检查一次，并未给出每个光纤段内的数值健康证据。GPU与CPU等价验证不能代替dz、dt和窗口收敛。应先定位首次谱边界增长所在段，再做更细dt/更小dz对照，不能简单放宽边界门限。</p>
<p>scan.aligned_residual实际比较时间对齐后的强度，属于r_I而非复场r_A；相位或啁啾变化可能被遗漏。当前地图是较早层级筛选，不是复场固定点认证。局部1.024 ns窗口只占约100 ns腔周期的1%，也不是全腔单脉冲证明。</p>
<p>历史g03_c11续算支持：该已测试初态在约705圈离开目标带、约1176圈出现多峰；更细网格继续到总2100圈仍未稳定。它否定了“第600圈已是稳定目标解”，却不能否定其他吸引域、更长时间后的转迁或附近参数。2100圈仍只有约210 μs。贴文“慢增益只是次要原因”的排序过早。</p>
<h2>6. 外部实验对当前模型的可比性</h2>
<p>Jeong等2014年论文确实报告9.80 MHz、12.7 ps、34 nJ的SWCNT锁模光纤激光器；说明相近重频和腔长可以实现CNT锁模，但并不直接证明当前器件配置在0.1–0.5 nJ稳定。其EDF色散约−22.3 ps²/km，而当前模型EDF为+45.9 ps²/km；DCF及SA实现也不同。净GDD相同不代表分段脉宽、CNT入射峰值或累积非线性相同。+0.141 ps²是有价值的取样参考，不是可直接移植的最优点。<a href="https://koasas.kaist.ac.kr/bitstream/10203/212673/1/96485.pdf">论文原文</a></p>
<p>2019年D形光纤SWCNT研究测得吸收体的偏振相关透射，并用较短腔实现低泵浦单孤子锁模。当前CNT模型是标量总功率吸收，PC只改变Jones态，另有固定双折射、Kerr及0.15 dB PDL；不能假定扫PC就复制那种器件的偏振相关非线性吸收。器件有效饱和能量还需要与光场重叠和模式面积相匹配。<a href="https://pdfs.semanticscholar.org/2fca/0aab562a2b81941c5387cd49ab83110c6c4d.pdf">论文原文</a></p>
<h2>7. 下一步验证应如何区分原因</h2>
<ol>
<li><b>先验证g08_c53能否形成稳定锚点。</b>沿真实时间延长，检查分段反转、瞬时平衡偏差、多个时间窗口的能量CV、复场残差和峰结构。预期若是慢收敛，慢变量与残差均下降；若出现持续周期，应报告周期呼吸；触边则先处理分辨率。不要预设50 mW末态已经收敛。</li>
<li><b>有锚点后再做双向支路追踪。</b>50→30 mW与30→50 mW以2 mW步长为初始安排，携带完整复场、纵向反转和CNT状态；另保留冷启动对照。每级满足收敛条件再前进，若达到时长上限只标“未收敛”。两方向不同提示滞回或吸引域问题，一条支路失稳不能证明所有支路无解。</li>
<li><b>独立检验CNT假说。</b>固定τ=0.5 ps、q₀、非饱和损耗及其他器件，令E_sat=5/10/20/40 pJ，对应P_sat=10/20/40/80 W。测实际q(t)、脉冲加权透射、扰动增长和稳态能量；加入与平均损耗匹配的线性吸收对照，区分“选择性增强”与“平均损耗降低”。变化支持机制后，再核实实物SA能否达到该参数。</li>
<li><b>最后细化色散及偏振。</b>优先+0.10–+0.20 ps²附近，但记录每次改变的SMF/DCF长度和非线性积分；固定总长的GDD扫描同时改变局部色散地图，并非只改变一个标量。PC扫描应保留当前模型与真实SA偏振响应差异的说明。</li>
</ol>
<p>若最后认证出1–2 nJ稳定输出，腔外衰减到0.1–0.5 nJ可以作为工程选项，但它不会改善原有相对能量波动或时序抖动。未认证的1.84 nJ状态不能通过加衰减器变成稳定种子。</p>
<h2>8. 当前判断的优先级</h2>
<table><tr><th>问题</th><th>证据等级与处理</th></tr><tr><td>慢增益未平衡</td><td>直接观测；优先解决观察时长与慢变量收敛</td></tr><tr><td>目标点CNT调制偏弱</td><td>直接观测；作为失稳原因尚需受控干预</td></tr><tr><td>高能量存在稳定单脉冲支路</td><td>值得验证，尚未建立</td></tr><tr><td>门槛太严格造成无候选</td><td>不能解释目标点19%–44%的CV</td></tr><tr><td>GPU实现是主要错误源</td><td>短程等价证据降低嫌疑；不能替代数值收敛与物理模型校验</td></tr><tr><td>整个目标能量区物理无解</td><td>现有证据不支持</td></tr></table>
<p>复现：diagnose_full_scan.py读取本轮原始MAT；mechanism_diagnostics.json区分第600圈与第601圈量。主地图、物理参数、历史报告及停止门槛均未改动。</p>
"""
)

page = (
    '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz锁模搜索：问题诊断</title><style>body{font:17px/1.8 "Microsoft YaHei",sans-serif;color:#20384c;background:#eef3f6;margin:0}main{max-width:1120px;margin:auto;background:white;padding:35px}h1{font-size:29px}h2{font-size:23px;margin-top:38px;border-left:5px solid #0088a5;padding-left:14px}aside{background:#fff4df;padding:20px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border:1px solid #d9e2e8;padding:10px;text-align:left}th{background:#eaf3f6}img{width:100%;height:auto}figure{margin:28px 0}figcaption{color:#567;font-size:15px}.scroll{overflow:auto}a{color:#087a9c}@media(max-width:600px){main{padding:15px}body{font-size:16px}}</style><main>'
    + body
    + "</main></html>"
)
(ROOT / "原生10MHz_锁模问题详细诊断.html").write_text(page, encoding="utf8")
print("Wrote standalone mechanism review")
