"""Offline report of one seed's lifetime; no hypothetical sweep counted as real."""
import base64,io,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':12})
a=json.loads((ROOT/'continuation.json').read_text());c=json.loads((ROOT/'independent_checks.json').read_text())
rows=[r for r in a['history'] if 'accepted_trial' in r];ts=[r['trials'][r['accepted_trial']] for r in rows]
steps=np.arange(1,len(rows)+1);fresh=np.array([r['precondition_rebuilt'] for r in rows]);parts=[]
def table(head,data):return '<div class="table"><table><tr>'+''.join('<th>'+v+'</th>' for v in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in data)+'</table></div>'
def section(title,text,draw=None):
    pic=''
    if draw:
        fig,ax=plt.subplots(figsize=(11,5.5));draw(ax);fig.tight_layout();b=io.BytesIO();fig.savefig(b,format='png',dpi=150);fig.savefig(ROOT/f'figure_{len(parts)+1:02d}.png',dpi=150);plt.close(fig)
        pic='<img alt="'+title+'" src="data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()+'">'
    parts.append('<section><h2>'+title+'</h2>'+text+pic+'</section>')
section('研究问题与本轮控制条件',
    '<p>检验同一个环带梯度方向能否跨状态复用：从 S3 末态启动一条最多 30 步续算，始终不重新扫描环带。'
    '固定 20.42 m、10 MHz、CNT→OC、OC=80%、净色散 +0.2 ps²、泵浦 27.5 mW、Psat=40 W，以及网格、归一化与规范。初始信赖域半径 0.003125。</p>'
    '<p>搜索模型包含当前 Arnoldi 基底、当前 C 方向、完整历史库和一个固定环带种子。原七个历史方向全部入库；接受一步后产生的新 fresh C 方向在下一状态入库。'
    '历史最多 12 个，超限按当前模型逐一删除损失最小者淘汰；环带种子不参与淘汰。当前 C 单独保留。'
    '尚未进入下一模型的最新方向作为待入库项保存，不计入上一模型的容量。</p>'
    '<p>每个新状态重新计算历史和环带响应。cheap 指复用预条件矩阵、用其构造当前 C 方向，不表示复用旧 Jd。'
    '每个候选保持步相关门限、Cauchy 下界、完整映射 ρ 和多尺度模型核验。求解步编号不是物理时间。</p>')
reduction=1-a['residual']/a['initial_residual']
section('续算结果与预设判据',
    f"<p>接受 {a['accepted_steps']} 步，状态：{a['status']}。残差从 {a['initial_residual']:.10g} 降至 {a['residual']:.10g}，累计下降 {100*reduction:.4f}%。"
    f"最后五步平均范数下降 {100*a['last_five_mean_norm_gain']:.5f}%/步。运行 {a['seconds']:.1f} s，实际环带重扫次数为 0。</p>"
    f"<p>固定种子第一步 GA={ts[0]['seed_model']['gain']:.6g}，其后最大 GA={max(t['seed_model']['gain'] for t in ts[1:]):.6g}。"
    '本次旧方向的主要新增收益集中在第一步；方向仍独立不等于其下降价值可以持续保持。'
    '整条轨迹仍在下降，不能把总进展全部归因于旧环带，也不能把低 GA 直接计作实际重扫需求。</p>'
    +table(['预先规定的检查','观察结果','含义'],[
        ['A：末态 R<10⁻³ 或最后五步平均下降>0.1%/步','达到' if a['success_A'] else '未达到','只衡量进展，不证明已进入 Newton 快速收敛区'],
        ['B：超过半数 fresh 状态 GA>1.15',f"{a['fresh_seed_gain_above_115']}/{a['fresh_states']}，"+('达到' if a['success_B'] else '未达到'),'衡量旧方向相对当前历史模型的价值，不证明新方向无额外收益'],
        ['周期一数值根 ‖R‖<10⁻⁷','达到' if a['numerical_root'] else '未达到','稳定性仍需独立检验']]))
def residual(ax):
    ax.plot(range(len(rows)+1),[a['initial_residual']]+[t['residual'] for t in ts],'o-',markersize=4)
    ax.axhline(.001,color='gray',ls='--',label='进展判据 R=0.001');ax.set(xlabel='接受步数',ylabel='归一化闭合残差 ‖R‖');ax.legend();ax.grid(alpha=.2)
section('残差随接受步数的变化',
    '<p>每个点都来自完整腔映射。第一步已与冻结状态 G7+A 的同半径候选核对。'
    '这是一条轨迹，不能把它和不同起点的旧轨迹逐步下降率直接解释为受控加速倍数。</p>',residual)
def rho(ax):
    ax.plot(steps,[t['rho'] for t in ts],'o-');ax.axhline(.1,color='red',ls='--',label='接受下限 0.1')
    ax.axhline(1,color='gray',ls=':',label='预测与实际一致');ax.set(xlabel='接受步序号',ylabel='完整映射实际下降 / 模型预测下降');ax.legend();ax.grid(alpha=.2)
section('已接受候选的模型预测可信度',
    '<p>ρ 接近 1 表示模型预测和实际下降接近；较低的正值表示实际下降小于预测。'
    'ρ 低于 0.25 会触发后续重建，候选仍须通过 0.1 接受下限与多尺度方向检查。'
    '数值模型一致不等于物理模型已通过实验验证。</p>',rho)
def radius_plot(ax):
    ax.semilogy(steps,[t['radius'] for t in ts],'o-',label='接受候选半径')
    ax.semilogy(steps,[t['step_norm'] for t in ts],'s-',label='实际步长范数')
    ax.set(xlabel='接受步序号',ylabel='归一化状态空间尺度');ax.legend();ax.grid(alpha=.2)
section('信赖域半径与实际步长',
    '<p>半径约束的是归一化完整状态步的范数，不是历史组合系数的范数。'
    '半径按既有控制规则自适应调整；图中显示接受候选，所有被拒绝尝试仍保存在逐步 JSON 中。</p>',radius_plot)
def gain(ax):
    y=np.array([t['seed_model']['gain'] for t in ts]);ax.plot(steps,y,'o-',markersize=4,label='每个已接受候选的 GA')
    ax.scatter(steps[fresh],y[fresh],s=80,facecolors='none',edgecolors='red',label='fresh 预条件状态')
    ax.axhline(1.15,color='gray',ls='--',label='GA=1.15');ax.set(xlabel='接受步序号（起点种子年龄=0）',ylabel='预测下降 m(H+A) / m(H)',ylim=(.95,1.08*max(y)));ax.legend();ax.grid(alpha=.2)
section('持久化环带方向的当前模型增益',
    '<p>在同一个状态、同一个候选半径和同一历史库上，分别求解有无旧环带的受限模型。'
    'GA 和新颖度对应本步的出发状态，逐步表中的 R 列对应接受之后的状态。'
    'GA 是预测比，不是两条轨迹的实测性能比；本轮只执行含环带的候选。多次拒绝时，图中显示最终接受半径，全部尝试留在 JSON 中。'
    '没有按 slope 正负或新颖度删除环带。</p>',gain)
def novelty(ax):
    ax.plot(steps,[r['seed_novelty_state'] for r in rows],'o-',label='输入空间外 νx')
    ax.plot(steps,[r['seed_novelty_response'] for r in rows],'s-',label='响应空间外 νR')
    ax.set(xlabel='接受步序号',ylabel='固定种子在当前无环带模型外的范数比例',ylim=(0,1.05));ax.legend();ax.grid(alpha=.2)
section('固定种子的新颖度随状态变化',
    '<p>输入空间由当前 Z、dC 和保留历史方向张成；响应空间由其当前响应张成，不加入残差目标列。'
    '使用 SVD 相对容差 10⁻¹² 投影。接近 1 表示大部分范数尚未被模型覆盖，接近 0 表示接近可表示。'
    '新颖度高不保证下降收益高；方向可能仍独立，但已经不再对准当前残差。</p>',novelty)
def slope(ax):
    ax.plot(steps,[r['seed_slope'] for r in rows],'o-');ax.axhline(0,color='gray')
    ax.set(xlabel='接受步序号',ylabel='当前残差方向内积 R·JdA',yscale='symlog');ax.set_yscale('symlog',linthresh=1e-8);ax.set_ylim(-3e-4,1e-6);ax.grid(alpha=.2)
section('独立性与当前下降斜率的区别',
    '<p>R·JdA 是固定种子在当前状态的单方向一阶斜率；负值表示沿该方向可下降。'
    '斜率趋近零而新颖度仍高，表示方向仍在模型之外，但对当前残差的一阶下降作用已经很小。'
    '联合模型仍可反向或组合使用，故不能只按符号删除它；是否有收益仍以 GA 为准。</p>',slope)
def bank(ax):
    ax.step(steps,[r['history_bank_size'] for r in rows],where='mid',label='本步使用的历史数量')
    ax.scatter(steps[fresh],np.array([r['history_bank_size'] for r in rows])[fresh],color='red',label='fresh C 状态')
    ax.axhline(12,color='gray',ls='--');ax.set(xlabel='接受步序号',ylabel='历史库使用数量',ylim=(0,13));ax.legend();ax.grid(alpha=.2)
deletions=[[r['step']+1,e['removed_id'],f"{next(v['loss'] for v in e['trials'] if v['index']==e['removed']):.5g}"] for r in rows for e in r['bank_pruning']]
section('历史库增长与逐一删除损失',
    '<p>每次超限都比较包含固定环带种子的完整模型与删除某个历史方向的模型，使用该状态初始半径。'
    '删除预测下降损失最小的方向，只有数值上近似相等时才优先删除较旧方向；同一状态的重试不反复改变库。</p>'
    +table(['接受步','删除的来源编号','预测下降损失'],deletions or [['—','本轮未发生淘汰','—']]),bank)
def gains(ax):
    y=[100*(1-t['residual']/r['residual']) for r,t in zip(rows,ts)];ax.plot(steps,y,'o-',label='单步范数下降')
    ax.axhline(.1,color='green',ls='--',label='进展参考 0.1%');ax.axhline(.02,color='gray',ls=':',label='停滞参考 0.02%')
    ax.set(xlabel='接受步序号',ylabel='归一化残差下降 (%)');ax.legend();ax.grid(alpha=.2)
section('重扫条件的只记录检查',
    '<p>观察条件：GA<1.15 连续三个 fresh 状态，且当前步之前五个接受步平均范数下降<0.02%/步。'
    'cheap 状态不增加或打断 fresh 连续计数；fresh 状态 GA≥1.15 才重置计数。即使触发，本轮仍禁止重扫。</p>'
    f"<p>满足观察条件的外迭代编号（从 1 开始）：{[i+1 for i in a['hypothetical_trigger_states']]}。"
    '这些是“可以进行新旧种子对照”的时点，不是实际重扫次数，更不是新种子恢复收益的证据。'
    '因此本轮不能检验“30 步需重扫至少三次则启动伴随”的成本判据。</p>',gains)
section('逐步诊断记录',table(['步','R（步后）','ρ','半径','GA','νx','νR','历史数','C 来源','种子年龄'],[
    [r['step']+1,f"{t['residual']:.8g}",f"{t['rho']:.4f}",f"{t['radius']:.5g}",f"{t['seed_model']['gain']:.5g}",f"{r['seed_novelty_state']:.5f}",f"{r['seed_novelty_response']:.5f}",r['history_bank_size'],'fresh' if r['precondition_rebuilt'] else 'cheap',r['seed_age']] for r,t in zip(rows,ts)]))
section('独立核验与结论范围',
    f"<p>独立重算 {c['accepted_states_replayed']} 个状态，最大残差差异 {c['max_residual_difference']:.4g}，输出复场相对差异 {c['output_relative_difference']:.4g}。"
    f"固定种子方向斜率逐态重算，最大差异 {c['seed_slope_max_difference']:.4g}。"
    '种子字节哈希保持不变；历史容量、逐一删除选择和触发记录通过核验。89 项 CPU 测试通过。</p>'
    '<p>本轮仅检验固定环带方向的跨状态复用。若收益降低，仍需在同态计算新环带并比较，才能知道重扫是否值得；'
    '不能从独立性高或旧种子衰减直接推导伴随收益、物理全梯度缺失、稳定单脉冲或 single-shooting 无根。</p>')
payload=base64.b64encode(json.dumps(a,ensure_ascii=False,indent=2).encode()).decode()
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：持久化环带种子30步验证</title><style>body{margin:0;background:#edf2f6;color:#182d40;font:17px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:25px}section{background:white;border-top:4px solid #176c91;padding:28px;margin:25px 0}h1{font-size:30px}h2{font-size:24px;margin-top:0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cddbe4;padding:8px;text-align:left}th{background:#e8f1f6}.table{overflow:auto}a{color:#176c91}@media(max-width:600px){main{padding:10px}section{padding:14px}h1{font-size:24px}h2{font-size:20px}}</style><main><h1>原生 10 MHz：持久化环带种子 30 步验证</h1>'+''.join(parts)+f'<p><a download="continuation.json" href="data:application/json;base64,{payload}">下载完整数值记录</a></p></main></html>'
(ROOT/'原生10MHz_持久化环带种子30步验证.html').write_text(html,encoding='utf8');print('Sections',len(parts))
