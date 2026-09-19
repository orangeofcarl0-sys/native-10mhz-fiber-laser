"""Self-contained Chinese descent-source report with separate scientific figures."""
import json,io,base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT,PROJECT
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':12})
a=json.loads((ROOT/'source_audit.json').read_text());parts=[]
def section(title,text,draw=None):
    image=''
    if draw:
        fig,ax=plt.subplots(figsize=(11,5.3));draw(ax);fig.tight_layout();buf=io.BytesIO();fig.savefig(buf,format='png',dpi=150);fig.savefig(ROOT/f'figure_{len(parts)+1:02d}.png',dpi=150);plt.close(fig)
        image='<img src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'">'
    parts.append('<section><h2>'+title+'</h2><p>'+text+'</p>'+image+'</section>')
section('研究问题与单状态对照','在上一轮 B 末态 R≈1.20814×10⁻³，检验新增方向能否提高当前受限步的实际下降。S1：当前 C 方向；S2：增加 375–750 GHz 环带方向；S3：增加最近三个新鲜历史方向；S4：增加最有利的物理参数坐标。共用当前 C 预条件器、完整网格上的 Arnoldi 基底、残差定义及三个半径。C 是显式梯度与预条件子空间，完整 Krylov 基底并未被硬截断到 C。S4 允许物理参数变化，其结论不代表原固定参数方程改善。')
section('比较规则与历史数据恢复','历史方向通过确定性重放上一段 30 步恢复，逐步残差及末态须匹配，随后在当前状态重新计算所有响应。每个候选须通过 Cauchy 模型下界、三个尺度的预测下降交叉检查（差异小于 5%）与完整映射下降比 ρ>0.1。新增组只有实际 merit 下降达到同半径有效 S1 的 1.25 倍，才有资格继续 30 步。此处 merit=‖R‖²/2。')
if (ROOT/'step_support.json').exists():
    support=json.loads((ROOT/'step_support.json').read_text())
    def step_bands(ax):
        for group in ['S1','S2','S3','S4']:
            rows=[row for row in support['partitions'] if row['group']==group]
            ax.plot([row['radius'] for row in rows],[100*row['annulus_field_fraction'] for row in rows],'o-',label=group)
        ax.set(xlabel='信赖域半径',ylabel='环带步长范数 / 全场步长范数 (%)');ax.legend();ax.grid(alpha=.2)
    section('预条件支持与实际搜索步的区别','预条件器在 C 外仍使用 −I，Arnoldi 向量因此可以覆盖完整网格。S1 本身已有非零环带分量；S3 的历史额外方向位于 C，但最终组合步也不只位于 C。图中是步长范数比例，不是光功率比例；各正交频带的平方范数分解已验证。',step_bands)
def cosine(ax):
    names=list(a['directions']);ax.bar(np.arange(len(names)),[a['directions'][v]['cosine'] for v in names]);ax.set_xticks(np.arange(len(names)),['当前 C','频谱环带','历史 1','历史 2','历史 3']);ax.set(ylabel='−R·(Jd) / (‖R‖‖Jd‖)');ax.axhline(0,color='gray');ax.grid(axis='y',alpha=.2)
section('审计起点残差与各方向响应的夹角','这里测量的是一个具体方向 d 的响应 Jd 与残差的夹角，不是残差向整个 Range(JE) 的投影。对下降方向，在无半径截断的线性模型中，最大相对 merit 下降等于该余弦的平方。小余弦不能单独证明整个输入子空间已达驻点。历史方向的余弦若为负，表示沿该方向当前上坡，但子空间仍可利用反向或组合。',cosine)
def gradients(ax):
    ax.bar(['C：中心差分','环带：中心差分'],[a['core_central_gradient_norm'],a['annulus']['gradient_norm']]);ax.set(ylabel='完整输出受限梯度范数',yscale='log');ax.grid(axis='y',alpha=.2)
section('核心频带与新增环带的梯度','这里的 GHz 是光场包络相对载频的傅里叶偏移频率，不是测距射频读出频点。两个频带均在原完整网格上计算完整残差内积。新增环带不包含反转或规范变量；流式计算中心差分及前向差分对照，不建立更大的稠密矩阵。梯度范数揭示一阶敏感度，是否有用还必须看受限模型和实际下降。',gradients)
def gradient_spectrum(ax):
    vectors=np.load(ROOT/'audit_vectors.npz');template=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
    n=template['original_template'].shape[-1];freq=np.fft.fftshift(np.fft.fftfreq(n,float(template['dt'])))*1000
    for key,norm,label in [('core',a['core_gradient_norm'],'当前 C 梯度'),('annulus',a['annulus']['gradient_norm'],'环带梯度')]:
        g=-vectors[key]*norm;field=(g[:2*n]+1j*g[2*n:4*n]).reshape(2,n)
        power=np.fft.fftshift(np.sum(abs(np.fft.fft(field,norm='ortho'))**2,axis=0))
        mask=abs(freq)<=375 if key=='core' else (abs(freq)>375)&(abs(freq)<=750)
        ax.semilogy(freq,np.where(mask,np.maximum(power,1e-30),np.nan),label=label)
    for cutoff in [-750,-375,375,750]:ax.axvline(cutoff,color='gray',ls='--',alpha=.4)
    ax.set(xlim=(-850,850),xlabel='光场包络频率偏移 (GHz)',ylabel='场梯度傅里叶系数平方和 / 频点');ax.legend();ax.grid(alpha=.2)
section('场梯度在频谱中的分布','图中是优化梯度的频谱，不是输出光功率谱。C 采用实际使用的前向流式梯度（已作中心差分复核），环带采用中心差分梯度；这里只画场坐标，不包含反转与规范分量。梯度较大不等于该频带光功率较大。',gradient_spectrum)
blocks=a['core_gradient_blocks']
section('残差占比与变量梯度的区别',f"审计起点 C 梯度的场、反转、规范分量范数分别为 {blocks['field']:.6g}、{blocks['population']:.6g}、{blocks['gauge']:.6g}。场残差占比高，并不意味着反转变量的梯度或耦合可忽略。核心梯度的前向/中心差分相对差异为 {100*a['core_gradient_stencil_difference']:.5g}%，环带为 {100*a['annulus']['forward_central_difference']:.5g}%，本轮未见梯度差分污染主导这些对照。")
def parameters(ax):
    names=list(a['parameters']);ax.bar(names,[a['parameters'][v]['bounded_prediction'] for v in names]);ax.set(ylabel='单参数模型下降（|Δq|≤0.003125）');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(axis='y',alpha=.2)
section('物理参数方向及尺度',f"所有列在 27.5 mW、GDD=0.2 ps²、OC=0.8、CNT Psat=40 W 处计算。归一化单位分别为 5 mW、0.02 ps²、0.05、4 W；GDD 保持总腔长 20.42 m，Psat 保持恢复时间。先按半径 0.003125 的一维模型下降选择 {a['selected_parameter']}，没有新增人为方程。这里的选择是按单参数模型排序，并非四个参数与 Krylov 联合后的全局最优比较。导数同时用 Δq=1e−3 与 5e−4 核对。",parameters)
table='<table><tr><th>参数</th><th>无约束线性最优 Δq</th><th>导数尺度差异</th></tr>'
for name,values in a['parameters'].items():
    table+=f"<tr><td>{name}</td><td>{values['free_scaled_step']:.5g}</td><td>{100*values['column_crosscheck']:.4g}%</td></tr>"
table+='</table>'
section('参数导数的一致性与线性步长', 'Δq 为上述归一化参数单位；无约束线性最优值只是局部模型诊断，不是已验证的可执行物理调参量。S4 候选另外受信赖域约束并运行完整模型。'+table)
def actual(ax):
    for i,g in enumerate(['S1','S2','S3','S4']):
        rows=[v for v in a['trials'] if v['group']==g];ax.bar(np.arange(3)+(i-1.5)*.19,[v['actual_reduction'] for v in rows],width=.19,label=g)
    ax.set_xticks(np.arange(3),[str(v) for v in a['radii']]);ax.set(xlabel='信赖域半径',ylabel='完整映射的实际 merit 下降');ax.legend();ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(axis='y',alpha=.2)
section('四组候选的实际单步下降','所有候选都从同一个保存态计算，图中没有串接前一个试步。模型下降与实际下降分别保留；候选未通过检查时，不能因柱形较高就宣称可用。',actual)
def gains(ax):
    for g in ['S2','S3','S4']:
        rows=[v for v in a['trials'] if v['group']==g and v['actual_gain_over_S1'] is not None];ax.plot([v['radius'] for v in rows],[v['actual_gain_over_S1'] for v in rows],'o-',label=g)
        bad=[v for v in rows if not v['passed']]
        if bad:ax.scatter([v['radius'] for v in bad],[v['actual_gain_over_S1'] for v in bad],marker='x',s=100,color='red')
    ax.axhline(1.25,ls='--',color='gray',label='续算门槛 1.25');ax.set(xlabel='信赖域半径',ylabel='实际下降 / 同半径 S1');ax.legend();ax.grid(alpha=.2)
section('新增方向的改善比例与续算门槛','红叉表示该候选未通过模型或实际下降检查。S1 实际下降非正时不定义改善比值，也不据此触发续算。必须同时满足物理可行、候选检查合格和至少 25% 的实际改善；不根据模型预测单独启动长程计算。',gains)
def rho(ax):
    for g in ['S1','S2','S3','S4']:
        rows=[v for v in a['trials'] if v['group']==g];ax.plot(a['radii'],[v['rho'] for v in rows],'o-',label=g)
    ax.axhline(.1,color='gray',ls='--');ax.set(xlabel='信赖域半径',ylabel='实际下降 / 模型下降 ρ');ax.legend();ax.grid(alpha=.2)
section('受限模型与实际映射的吻合','实际下降比检验线性模型在该半径是否可信。接近 1 说明预测吻合，不等于下降幅度大，也不等于即将收敛。',rho)
qualified=a['qualified'];section('对照结果与解释边界','达到预设续算门槛的候选：'+('；'.join(f"{v['group']}，Δ={v['radius']}，实际改善 {v['gain']:.3f} 倍" for v in qualified) if qualified else '无。按约定不启动新的 30 步续算。')+'。这些结果只覆盖指定频带、三个历史方向和四个局部参数方向；未审计的更高频段和其他方向仍存在；不能证明全空间驻点或不存在周期一根。下一步方案须根据实际对照结果决定。')
section('单状态结果的含义','环带梯度约为 C 梯度的 5.96 倍，提供了有效的新方向；但仍位于原 C 支持内的历史方向也得到接近的实际收益。因此不能把问题单独归因于频谱带宽，也不能用当前单方向的小夹角断言 C 内已无有效下降。S4 选中 Psat；其无约束线性最优 Δq≈0.141 超出所测半径，当前比较不排除更大范围的参数调整。')
if (ROOT/'continuation.json').exists():
    b=json.loads((ROOT/'continuation.json').read_text());accepted=[row for row in b['history'] if 'accepted_trial' in row]
    values=[b['initial_residual']]+[row['trials'][row['accepted_trial']]['residual'] for row in accepted]
    def trace(ax):
        ax.plot(range(len(values)),values,'o-');ax.set(xlabel='S3 本轮接受步',ylabel='周期一缩放残差范数');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(alpha=.2)
    section('S3 多步续算结果',f"S3 最优单步实际下降达到全组最佳的约 99.6%，且无须重复流式扫描环带，因此选择它续算。采用滚动的最近三个新鲜历史方向，响应逐步重算。实际接受 {b['accepted_steps']} 步，残差 {b['initial_residual']:.9g} → {b['residual']:.9g}，下降 {100*(1-b['residual']/b['initial_residual']):.4f}%；停止状态 { {'iteration_budget_reached':'完成本轮迭代预算','residual_converged':'达到数值求根门限'}.get(b['status'],b['status']) }。求根门限仍为 1e−7。S2 尚未进行同预算长程比较。",trace)
    def rates(ax):
        arr=np.array(values);ax.plot(range(1,len(arr)),100*(1-arr[1:]/arr[:-1]),'o-');ax.set(xlabel='S3 本轮接受步',ylabel='单步残差相对下降 (%)');ax.grid(alpha=.2)
    section('S3 逐步改善与持续性','观察新增历史信息是否带来持续改善。单状态的大倍数提升不等于多步收敛速度提升相同倍数，更不等于达到周期一根或稳定性认证。',rates)
    if (ROOT/'endpoint_checks.json').exists():
        verified=json.loads((ROOT/'endpoint_checks.json').read_text());arr=np.array(values);last5=float(np.mean((100*(1-arr[1:]/arr[:-1]))[-5:]))
        section('多步验证与结论边界',f"独立残差重算差 {verified['endpoint_difference']:.1g}，输出光场差 {verified['output_difference']:.1g}；所有接受条件通过，回退 {verified['fallbacks']} 次。新鲜方向 {verified['fresh']} 次、经过当前响应检查的廉价方向 {verified['cheap']} 次。末 5 步平均改善 {last5:.5f}%/步，仍未达到数值根。末态场残差 {verified['residual_blocks']['field']:.6g}、反转残差 {verified['residual_blocks']['population']:.6g}；反转残差较起点有所增加，不能说所有块都改善。候选输出能量 {verified['output_energy_nJ']:.6f} nJ，未作稳定单脉冲认证。")
elif (ROOT/'continuation_protocol.json').exists():
    section('已启动的后续验证','S3 的最佳单步实际下降达到全组最佳的约 99.6%，且后续成本较低，已按门槛启动 30 步续算。当前多步结果尚未完成；本页单状态对照已完成。')
links=''
for name in ['source_audit.json','history_recovery.json','protocol.json','continuation.json','endpoint_checks.json','continuation_protocol.json','candidate_replay.json','step_support.json']:
    if (ROOT/name).exists():links+=f'<a download="{name}" href="data:application/json;base64,{base64.b64encode((ROOT/name).read_bytes()).decode()}">{name}</a> '
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：下降方向来源对照</title><style>body{margin:auto;max-width:1200px;padding:24px;background:#eef3f8;color:#17344a;font:18px/1.7 system-ui}section{padding:30px;margin:24px 0;background:white;border-radius:12px}h1,h2{color:#065783}img{width:100%;height:auto}a{overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:16px}td,th{border-bottom:1px solid #ccd9e5;padding:8px;text-align:left}@media(max-width:600px){body{padding:10px}section{padding:15px}h1{font-size:25px}h2{font-size:21px}}</style><h1>原生10MHz：下降方向来源的单状态四组对照</h1>'+links+''.join(parts)+'</html>'
(ROOT/'原生10MHz_下降方向来源与四组对照.html').write_text(html,encoding='utf8')
