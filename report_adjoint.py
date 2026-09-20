"""Standalone Chinese report; plot numerical validation before solver effects."""
import base64,io,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':12})
validation=json.loads((ROOT/'adjoint_validation.json').read_text())
data=json.loads((ROOT/'adjoint_frozen.json').read_text())
checks=json.loads((ROOT/'independent_checks.json').read_text())
rows=data['states'];xx=np.arange(len(rows));labels=[str(s['step']) for s in rows];sections=[]


def table(headers,rows):
    return '<div class="table"><table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</table></div>'


def section(title,body,draw=None):
    img=''
    if draw:
        fig,ax=plt.subplots(figsize=(11,5.5));draw(ax);fig.tight_layout();buf=io.BytesIO()
        fig.savefig(buf,format='png',dpi=150);fig.savefig(ROOT/f'figure_{len(sections)+1:02d}.png',dpi=150);plt.close(fig)
        img='<img alt="'+title+'" src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'">'
    sections.append('<section><h2>'+title+'</h2>'+body+img+'</section>')


section('研究问题与本轮范围',
 '<p>上一轮四状态证明 fresh 环带梯度能恢复收益。本轮检验：完整离散伴随能否准确、低成本地产生当前梯度；以及单个 full-gradient 方向是否比 fresh annulus 更有效。</p>'
 '<p>实现通用 <code>CavityResidual.vjp(x,v)</code>，返回经过状态缩放的实向量 Jᵀv；首次应用为 v=R。固定 20.42 m、10 MHz、CNT→OC、OC=80%、净色散 +0.2 ps²、27.5 mW 泵浦和原网格。'
 '沿用保存状态 5、11、20、27（零起算），不做新轨迹、不改重扫策略、GMRES、Hookstep 或历史库策略。</p>'
 f'<p>四状态 annulus 投影相对误差最大 {max(s["annulus_relative"] for s in validation["states"]):.3g}；全部完整残差分块测试通过。'
 f'本设备测得 streaming/伴随加速 {min(s["speedup"] for s in rows):.1f}–{max(s["speedup"] for s in rows):.1f} 倍；'
 f'full 候选在 {sum(s["full_to_annulus_actual"]>1 and s["trials"][2]["passed"] for s in rows)}/4 个状态取得比 annulus 更大的真实下降且通过验收。详细数值与边界见后文。</p>')
section('实际反传的数学对象',
 '<p>输入为 x=(Re(a)/s, Im(a)/s, p/√Nc, φ, Δt/ts)。场残差为 [exp(−iφ)SΔtF(a,p)−a]/s；反转残差为 [p−A/B]/√Nc，另有两个固定模板规范条件。'
 '因此必须反传相位、时移、场的直接负项，以及 population 与光场的双向耦合。</p>'
 +table(['层','实现','必须保留的依赖','核验'],[
 ['A 外壳','adjoint_primitives.shell_vjp','缩放、相位、时移、population、两个 gauge','解析假线性腔 tangent'],
 ['B 线性光学','discrete_adjoint.segment_vjp','FFT 共轭、Jones 共轭转置、OC/损耗/PDL','线性点积恒等式'],
 ['C Kerr','adjoint_primitives.kerr_vjp','两圆偏振强度与交叉相位','中心差分尺度扫描'],
 ['D CNT','adjoint_primitives.cnt_vjp','中点透射与全时间递推','早输入→晚输出测试'],
 ['E EDF','discrete_adjoint.segment_vjp','冻结反转增益、A/B、信号功率和沿程泵','纯 population cotangent 测试']])
 +'<p>监测输出 out、输出能量、末端 CNT q、sa/last_gap/last_tau 和末端 pump 不进入目标，输出 cotangent 为零；内部 pump 递推仍影响后续 cell，不能删除。</p>')


def primitive_plot(ax):
    p=checks['primitives']
    for name,values in p['relative_errors'].items():ax.loglog(p['scales'],np.maximum(values,1e-16),'o-',label=name)
    ax.axhline(1e-5,color='gray',ls='--');ax.set(xlabel='中心差分步长 h',ylabel='dot-product 相对误差');ax.legend();ax.grid(alpha=.2)
section('局部算子的差分尺度验证',
 '<p>点积检查比较 (Jhu)·v 与 u·(Jᵀv)。非线性算子使用五个 h，保留整条误差曲线；验收为至少一个尺度低于 10⁻⁵，并审查尺度变化。'
 '线性光学与解析外壳单元测试要求低于 10⁻¹²。小 h 的差分舍入误差不应解释为伴随模型误差。</p>',primitive_plot)
blockrows=[]
for s in validation['states']:
    for d in s['dots']:blockrows.append([s['step'],d['kind'],f"{min(t['relative'] for t in d['scales']):.3g}",f"{max(t['relative'] for t in d['scales']):.3g}"])
section('完整残差的分块 dot-product 验证',
 '<p>每个真实状态分别检查 full、field-only u、population-only u、phase/time-only u，以及 field-only、population-only、gauge-only v。'
 '这样能揭示块间遗漏，避免只测 JᵀR 时误差相互抵消。完整五尺度记录可从本报告下载。</p>'+table(['step','输入/输出块','最佳相对误差','五尺度最大误差'],blockrows))
section('四状态与 streaming 梯度的对照',
 '<p>保存的 annulus 单位方向与梯度范数共同恢复原 6144 列中心差分梯度；本轮还重新扫描该环带用于同口径计时。'
 'core C 则对照原预条件器重建时使用的 full-output 前向差分梯度，差分格式与 annulus 不同。比较方向同时比较范数、斜率和 Cauchy 步。</p>'
 +table(['step','annulus 相对误差','core 相对误差','CPU/GPU JᵀR 差异','αC','ΔmC'],[
 [s['step'],f"{s['annulus_relative']:.3g}",f"{r['core_relative']:.3g}",f"{s['parity']['vjp']:.3g}",f"{s['alpha_C']:.5g}",f"{s['prediction_C']:.5g}"] for s,r in zip(validation['states'],rows)])
 +'<p>环带单位下降方向 d 的斜率为 −R·Jd；αC=min(固定半径, −R·Jd/‖Jd‖²) 是缩放状态坐标中的 Cauchy 步长，不是距离或物理时间。ΔmC 为该步的预测 merit 下降。</p>'
 +table(['step','伴随环带梯度范数','与 streaming 的方向余弦','−R·Jd'],[
 [s['step'],f"{s['annulus_norm']:.8g}",f"{s['annulus_cosine']:.12f}",f"{s['slope']:.8g}"] for s in validation['states']]))


def timing_plot(ax):
    for i,(key,name) in enumerate([('forward','普通 residual'),('adjoint','正向 + VJP'),('streaming','6144 列 streaming')]):
        ax.bar(xx+(i-1)*.25,[s[key]['median_seconds'] for s in rows],.25,label=name)
    ax.set_yscale('log');ax.set_xticks(xx,labels);ax.set(xlabel='保存状态 step',ylabel='同步墙钟时间 / s');ax.legend();ax.grid(axis='y',alpha=.2)
section('同状态 GPU 时间与内存成本',
 f'<p>设备：{checks["hardware"]["gpu"]}，CuPy {checks["hardware"]["cupy"]}。预热后同步 GPU 计时：普通 residual 与正向+VJP 各测五次取中位数，每点 streaming 测完整一次。伴随包含构建 primal residual 和局部 replay。'
 'S=Tstream/Tadj；Cadj=Tadj/Tforward。单次 streaming 没有重复测量区间，速度倍数只代表本设备和本次条件。</p>'
 +table(['step','forward / s','adjoint / s','streaming / s','S','Cadj','伴随/stream 池保留峰值 MiB'],[
 [s['step'],f"{s['forward']['median_seconds']:.4f}",f"{s['adjoint']['median_seconds']:.4f}",f"{s['streaming']['median_seconds']:.2f}",f"{s['speedup']:.1f}",f"{s['forward_cost']:.2f}",f"{s['adjoint']['peak_reserved_bytes']/2**20:.1f} / {s['streaming']['peak_reserved_bytes']/2**20:.1f}"] for s in rows])
 +'<p>内存口径：每次调用采用独立 CuPy 内存池，记录调用结束仍保留的分配总量（期间不清池），作为池内 live 张量峰值的上界；不含 CUDA 上下文、预先存在的数组与 FFT plan，因此不是整进程峰值。'
 '实现保存六段入口，反向时只重放并暂存当前段；CNT 保存少量标量序列。</p>',timing_plot)


def gains(ax):
    for i,name in enumerate(['history','annulus','full']):ax.bar(xx+(i-1)*.25,[s['trials'][i]['gain'] for s in rows],.25,label=name)
    ax.set_xticks(xx,labels);ax.set(xlabel='保存状态 step',ylabel='相对固定 H 的预测下降倍数');ax.legend();ax.grid(axis='y',alpha=.2)
section('固定模型中的三个梯度方案',
 '<p>H 为原 Arnoldi 基底、当前 C 方向和原历史库。对照 H、H+fresh annulus、H+归一化 full gradient；三者使用同一半径。'
 '原 history 与 annulus 结果先恢复一致，才能解释 full 的差异。这里的候选不写回保存轨迹。</p>'
 +table(['step','annulus/H','full/H','full/annulus 预测','full/annulus 实际'],[
 [s['step'],f"{s['trials'][1]['gain']:.6g}",f"{s['trials'][2]['gain']:.6g}",f"{s['full_to_annulus_prediction']:.6g}",f"{s['full_to_annulus_actual']:.6g}"] for s in rows]),gains)


def actual(ax):
    for i,name in enumerate(['history','annulus','full']):ax.bar(xx+(i-1)*.25,[s['trials'][i]['actual'] for s in rows],.25,label=name)
    ax.set_xticks(xx,labels);ax.set(xlabel='保存状态 step',ylabel='完整 residual 的实际 merit 下降');ax.legend();ax.grid(axis='y',alpha=.2)
section('真实映射与候选验收',
 '<p>实际下降=[‖R(x)‖²−‖R(x+s)‖²]/2，ρ=实际/预测。检查可行性、半径、Cauchy 下界、三尺度方向一致性、正下降与ρ&gt;0.1。'
 '独立脚本重新核算保存的 12 个候选和压缩模型，不仅复用报告数值。</p>'+table(['step','候选','预测','实际','ρ','通过'],[
 [s['step'],t['name'],f"{t['prediction']:.5g}",f"{t['actual']:.5g}",f"{t['rho']:.5g}",t['passed']] for s in rows for t in s['trials']]),actual)
section('结论与接回求解器前的边界',
 '<p>本轮建立了通用离散 VJP，并用局部算子、任意 cotangent 分块、四状态 streaming 和真实候选四层证据验证。性能收益与候选数学收益必须分别判断。</p>'
 '<p>full direction 即使优于 annulus，也不能单凭这一个对照定位到 ±750 GHz 以外：full 同时含 core、population 和规范变量。若效果相近，主要收益可以是减少梯度计算成本；若更差，可用伴随投影得到 annulus 方向，保留已有子空间策略。</p>'
 '<p>本轮仅增加通用 API，现有求解器尚未自动改用它；没有新的 30 步续算，没有 JᵀJ Krylov、LSMR 或 Gauss–Newton。当前证据不构成物理根或稳定单脉冲认证。</p>')
payload=base64.b64encode(json.dumps(dict(validation=validation,experiment=data,checks=checks),ensure_ascii=False).encode()).decode()
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：离散伴随验证</title><style>body{margin:0;background:#edf2f6;color:#182d40;font:17px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:25px}section{background:white;border-top:4px solid #176c91;padding:28px;margin:25px 0}h1{font-size:30px}h2{font-size:24px;margin-top:0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cddbe4;padding:8px;text-align:left}th{background:#e8f1f6}.table{overflow:auto}a{color:#176c91}@media(max-width:600px){main{padding:10px}section{padding:14px}h1{font-size:24px}h2{font-size:20px}}</style><main><h1>原生 10 MHz：离散伴随与四状态固定步验证</h1>'+''.join(sections)+f'<p><a download="adjoint_records.json" href="data:application/json;base64,{payload}">下载全部数值记录</a></p></main></html>'
(ROOT/'原生10MHz_离散伴随与四状态固定步验证.html').write_text(html,encoding='utf8')
print('Report sections',len(sections))
