"""Offline report of the single controlled forty-step trajectory."""
import base64,io,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':12})
a=json.loads((ROOT/'continuation.json').read_text());checks=json.loads((ROOT/'independent_checks.json').read_text())
rows=[r for r in a['history'] if 'accepted_trial' in r];trials=[r['trials'][r['accepted_trial']] for r in rows]
x=np.arange(1,len(rows)+1);norms=[a['initial_residual']]+[t['residual'] for t in trials]
gains=np.array([1-t['residual']/r['residual'] for r,t in zip(rows,trials)])
sections=[]
def table(head,rows):
    return '<div class="table"><table><tr>'+''.join('<th>'+h+'</th>' for h in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
def section(title,body,draw=None):
    image=''
    if draw:
        fig,ax=plt.subplots(figsize=(11,5.5));draw(ax);fig.tight_layout();b=io.BytesIO()
        fig.savefig(b,format='png',dpi=150);fig.savefig(ROOT/f'figure_{len(sections)+1:02d}.png',dpi=150);plt.close(fig)
        image='<img alt="'+title+'" src="data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()+'">'
    sections.append('<section><h2>'+title+'</h2>'+body+image+'</section>')
mean=a['last_five_mean_norm_gain'];mg=a['last_five_mean_full_gain']
regime_label={'improved':'明显改善区间','sustained':'持续推进区间','plateau':'平台判据成立','intermediate':'中间区间（未满足明显改善或平台判据）','insufficient_steps':'步数不足'}[a['regime']]
section('受控续算的起点与结果',
 '<p>本轮只检验 full adjoint gradient 接回现有 history-bank + GMRES/Hookstep 后，能否跨过此前的慢平台。'
 '从最新末态开始，继承半径 0.00625 和 12 个旧历史方向；不退回四个回溯状态。固定泵浦 27.5 mW、OC=80%、GDD=+0.2 ps²、Psat=40 W、20.42 m、10 MHz、网格与规范模板。</p>'
 f'<p>计划 40 个 outer iteration，实际接受 {a["accepted_steps"]} 步，停止状态：{a["status"]}。'
 f'残差从 {a["initial_residual"]:.10g} 降至 <b>{a["residual"]:.10g}</b>，累计下降 {100*(1-a["residual"]/a["initial_residual"]):.3f}%。'
 f'R&lt;10⁻³：{"达到" if a["strong_success"] else "未达到"}；末五步平均下降 {100*mean:.5f}%/步，{regime_label}。</p>')
section('每步使用的梯度和历史库',
 '<p>每个 outer iteration 计算 (R,g)=value_and_vjp(x)，取 d=−g/‖g‖。C 只生成预条件器，搜索空间为 span{Z,d,Dhist}；不再计算 C-gradient 或 annulus gradient，也不保留 annulus seed。</p>'
 '<p>每个接受步的 full direction 都进入下一状态的候选历史。模型内最多 12 个历史方向，超额时按包含当前 d 的模型 leave-one-out 损失淘汰。末态待处理库可能暂含刚加入的第 13 个方向，下个模型使用前再按同一规则裁剪。</p>'
 '<p>Full-gradient Cauchy 的步长为 min(信赖半径, −R·Jd/‖Jd‖²)。增强步仍须不劣于此预测；保留 step-relevant gate、多尺度 Js、非线性 ρ、可行性和 Cauchy fallback。</p>')
def residual(ax):
    ax.plot(np.arange(len(norms)),norms,'o-',ms=3);ax.axhline(.001,color='red',ls='--',label='预设强成功线 0.001')
    ax.set(xlabel='接受步数',ylabel='完整残差范数');ax.legend();ax.grid(alpha=.2)
section('完整残差沿正式轨迹的变化',
 '<p>横轴是求解器接受步数，不是物理往返时间。每个点来自同一条正式轨迹，终点未经过参数调整或人为筛选。</p>',residual)
def progress(ax):
    ax.plot(x,100*gains,'o-',ms=3,label='单步相对下降')
    if len(x)>=5:ax.plot(x[4:],100*np.convolve(gains,np.ones(5)/5,'valid'),label='五步平均')
    for val in [.2,.05,.02]:ax.axhline(val,ls='--',alpha=.5,label=f'{val}%/步')
    ax.set(xlabel='接受步数',ylabel='残差相对下降 / %');ax.legend();ax.grid(alpha=.2)
section('末段推进速度与预设判据',
 f'<p>末五步平均相对下降为 {100*mean:.5f}%/步，平均 Gfull/H={mg:.6g}。'
 '预先规定：&gt;0.2% 为 improved；0.05%–0.2% 为 sustained；&lt;0.02% 且末五步平均 Gfull/H≤1.1 为 plateau；其余区间为 intermediate。'
 '没有根据结果移动阈值，也没有因下降变慢而提前停机。</p>',progress)
def blocks(ax):
    for key in ['field','population','gauge']:
        vals=[r['residual_blocks'][key] for r in rows]+[a['residual_blocks'][key]]
        ax.semilogy(np.arange(len(vals)),np.maximum(vals,1e-30),label=key)
    ax.set(xlabel='接受状态编号',ylabel='残差块范数');ax.legend();ax.grid(alpha=.2)
section('场闭合与反转、规范残差',
 f'<p>终点 field={a["residual_blocks"]["field"]:.6g}，population={a["residual_blocks"]["population"]:.6g}，gauge={a["residual_blocks"]["gauge"]:.6g}。'
 f'场项占完整残差平方范数的 {100*(a["residual_blocks"]["field"]/a["residual"])**2:.6f}%。'
 '分块检查用于区分场闭合、慢反转和规范条件的影响，不从总残差单独推断瓶颈。</p>',blocks)
def gradients(ax):
    ax.semilogy(x,[r['gradient_norm'] for r in rows],'o-',ms=3,label=r'$\|J^T R\|$')
    ax.set(xlabel='接受步数',ylabel='完整梯度范数');ax.legend();ax.grid(alpha=.2)
section('当前 full gradient 的变化',
 '<p>梯度在每个 outer 状态重新计算，包括复用 C 预条件器的步。梯度范数属于当前缩放状态度量，不能直接解释为脉冲能量或物理误差。</p>',gradients)
def coverage(ax):
    for name in ['raw','kept']:
        ax.plot(x,[next(s['coverage'] for s in r['span'] if s['span']==name) for r in rows],label=name)
    ax.set(xlabel='接受步数',ylabel='GMRES 空间对单位 full direction 的投影范数',ylim=(0,1.02));ax.legend();ax.grid(alpha=.2)
section('GMRES 空间的梯度覆盖率',
 '<p>coverage=‖ProjZ d‖/‖d‖，是范数比例而非平方能量比例。raw/kept 分别采用既有 10⁻¹⁴/10⁻¹² 相对 SVD 截断；coverage 低说明当前 root Krylov 空间尚未覆盖完整下降方向。</p>',coverage)
def fullgain(ax):
    ax.plot(x,[t['gradient_model']['gain'] for t in trials],'o-',ms=3)
    ax.axhline(1.1,ls='--',color='gray',label='平台诊断参考 1.1');ax.set(xlabel='接受步数',ylabel='Gfull/H');ax.legend();ax.grid(alpha=.2)
section('当前梯度的边际模型增益',
 '<p>Gfull/H=Δm(Z+Dhist+d)/Δm(Z+Dhist)，使用每步最终接受半径。分母不含当前 full direction，也不再单列 current C-gradient，因此不能直接与上一轮含 current C 的四状态倍率比较。若实际使用 fallback，表中的增强模型增益仍为模型诊断，实际步类型另行记录。</p>',fullgain)
def composition(ax):
    old=np.array([r['old_history_count'] for r in rows]);new=np.array([r['full_history_count'] for r in rows])
    ax.bar(x,old,label='继承的旧历史');ax.bar(x,new,bottom=old,label='本轮 full-gradient 历史')
    ax.set(xlabel='接受步数',ylabel='实际模型历史方向数',ylim=(0,13));ax.legend(loc='upper center',bbox_to_anchor=(.5,1.16),ncol=2)
section('历史库组成与替换',
 '<p>旧方向没有一次性清空。颜色表示每个实际模型中的新旧方向数量；当前 d 单独加入，不计入 12 个历史方向。</p>',composition)
section('保护检查与完整逐步记录',
 f'<p>独立复算 {checks["states_replayed"]} 个状态，核对 {checks["current_full_directions_verified"]} 个当前 full direction、Cauchy 候选、真实下降和淘汰规则。'
 f'预条件器重建 {sum(r["precondition_rebuilt"] for r in rows)} 次，fallback 接受 {sum("fallback" in r["selected_kind"] for r in rows)} 次，未接受尝试 {sum(len(r["trials"]) for r in a["history"])-len(rows)} 次；运行时间 {a["seconds"]:.1f} s。重建可能由同一步较早的未接受尝试触发，不能只看最终接受行的 ρ。</p>'
 +table(['步','R','Δ','ρ','Cauchy预测','Gfull/H','full/旧历史','步类型'],[
 [i+1,f"{t['residual']:.7g}",t['radius'],f"{t['rho']:.4g}",f"{t['cauchy_prediction']:.4g}",f"{t['gradient_model']['gain']:.4g}",f"{r['full_history_count']}/{r['old_history_count']}",t['kind']] for i,(r,t) in enumerate(zip(rows,trials))]))
section('本轮结论的范围',
 f'<p>本轮结果：{"已跨过" if a["strong_success"] else "尚未跨过"} 10⁻³ 强成功线；末段属于{regime_label}，平均 full/H={mg:.5g}。'
 '残差速度与当前梯度的边际增益必须分别解释，不能把慢推进直接等同于梯度失效。</p>'
 '<p>本轮给出的是 full adjoint + history + GMRES/Hookstep 的多步验证。以上指标不等于已到 10⁻⁷，也不构成稳定单脉冲认证。</p>'
 '<p>没有增加 Gauss–Newton、multiple shooting、参数释放、period-2 或 Floquet。下一步应依据末段速度及 full-gradient 增益判断，不能仅凭局部一两步下降决定再升级算法。</p>')
payload=base64.b64encode(json.dumps(dict(result=a,checks=checks),ensure_ascii=False).encode()).decode()
style='body{margin:0;background:#edf2f6;color:#182d40;font:17px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:25px}section{background:white;border-top:4px solid #176c91;padding:28px;margin:25px 0}h1{font-size:30px}h2{font-size:24px;margin-top:0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cddbe4;padding:8px;text-align:left}th{background:#e8f1f6}.table{overflow:auto}a{color:#176c91}@media(max-width:600px){main{padding:10px}section{padding:14px}h1{font-size:24px}h2{font-size:20px}}'
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：伴随梯度40步续算</title><style>'+style+'</style><main><h1>原生 10 MHz：伴随梯度与历史库正式续算</h1>'+''.join(sections)+f'<p><a download="continuation.json" href="data:application/json;base64,{payload}">下载完整记录</a></p></main></html>'
(ROOT/'原生10MHz_伴随梯度与历史库40步续算.html').write_text(html,encoding='utf8');print('Sections',len(sections))
