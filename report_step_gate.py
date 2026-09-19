"""Offline report for the step gate audit and conditional continuation."""
import base64,json,io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':13})
a=json.loads((ROOT/'audit.json').read_text());b=json.loads((ROOT/'continuation.json').read_text()) if (ROOT/'continuation.json').exists() else None
sections=[]
def section(title,text,plot=None):
    img=''
    if plot:
        fig,ax=plt.subplots(figsize=(11,5.4));plot(ax);fig.tight_layout();buf=io.BytesIO();fig.savefig(buf,format='png',dpi=150);fig.savefig(ROOT/f'figure_{len(sections)+1:02d}.png',dpi=150);plt.close(fig)
        img='<img src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'">'
    sections.append('<section><h2>'+title+'</h2><p>'+text+'</p>'+img+'</section>')
section('实验范围与判据','固定 27.5 mW 泵浦、20.42 m 总腔长、CNT → OC、80% 输出耦合、净色散 +0.2 ps²，沿用上一轮状态、缩放和规范。先检查保存的 A/B 末态，再有条件地从 B 末态续算。无泵浦扫描，无新增稳定锁模认证。残差是缩放后的周期一闭合误差，不是输出功率误差。')
def plateau(ax):
    for arm,v in a['arms'].items():ax.loglog(a['scales'],[r['eta'] for r in v['rows']],'o-',label=arm+'：直接方向差分')
    ax.axhline(.01,color='gray',ls='--',label='原 Newton 门限 1%');ax.set(xlabel='归一化状态扰动 h',ylabel='‖R + D_h(d_N)‖ / ‖R‖');ax.legend();ax.grid(alpha=.25)
section('巨大 Newton 方向的差分尺度敏感性','按单位方向扰动 x ± h·d/‖d‖，再将差分响应乘回 ‖d‖。A 在 h=1e−5 时误差约 0.913%，在 1e−6 时升至 3.009%；B 在中间尺度仍约 1.74%，高于 1%。这区分了小步长消减误差与方向本身的线性不一致。不能把最小步长当成真值。',plateau)
def rich(ax):
    for arm,v in a['arms'].items():ax.loglog(a['scales'],[r['richardson_eta'] for r in v['rows']],'o-',label=arm+'：Richardson')
    ax.set(xlabel='中心差分 h（同时计算 h/2）',ylabel='Richardson 方向检查误差');ax.legend();ax.grid(alpha=.25)
section('Richardson 外推与 Arnoldi 内部残差','外推为 [4D(h/2)−D(h)]/3，仅作为尺度诊断。Arnoldi 内部残差衡量基向量响应的线性组合；直接对最终巨大方向作差分是另一项检查。有限差分并非严格线性算子，内部残差接近零不代表直接方向检查也接近零。',rich)
def candidates(ax):
    for kind in ['cauchy','raw_augmented']:
        rows=[v for v in a['candidates'] if v['kind']==kind];ax.plot([v['radius'] for v in rows],[v['rho'] for v in rows],'o-',label={'cauchy':'Cauchy 下降步','raw_augmented':'原始增广步'}[kind])
    ax.axhline(.1,color='gray',ls='--');ax.set(xlabel='信赖域半径 Δ',ylabel='实际下降 / 模型预测下降 ρ');ax.legend();ax.grid(alpha=.25)
section('B 末态受限候选的完整映射检验','每个半径分别检查 Cauchy 步和原始增广步，状态保持不变。三个原始增广步的实际下降比约为 0.977、0.917、0.619，模型差异约 0.049%。原始增广模型须不差于 Cauchy；候选模型与独立方向差分须相差小于 5%，实际下降比须大于 0.1。',candidates)
def discrepancies(ax):
    for v in a['candidates']:
        if v['kind']=='raw_augmented':ax.loglog(a['scales'],[max(c['discrepancy'],1e-12) for c in v['checks']],'o-',label='Δ='+str(v['radius']))
    ax.axhline(.05,color='gray',ls='--',label='5%');ax.set(xlabel='候选步方向检查 h',ylabel='预测下降的相对差异');ax.legend();ax.grid(alpha=.25)
section('候选步模型的一致性','这些检查针对真正执行的小步；巨大 Newton 方向的检查仍保存为诊断。正式续算仅改变门限对象，Arnoldi 的 0.8% 内部目标和最多 240 维保持不变。',discrepancies)
if b:
    accepted=[row for row in b['history'] if 'accepted_trial' in row]
    vals=[b['initial_residual']]+[row['trials'][row['accepted_trial']]['residual'] for row in accepted]
    def convergence(ax):
        ax.plot(range(len(vals)),vals,'o-');ax.set(xlabel='本轮累计接受步',ylabel='周期一缩放残差范数');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(alpha=.25)
    section('从 B 检查点续算的残差变化',f"接受 {b['accepted_steps']} 步；残差 {b['initial_residual']:.9g} → {b['residual']:.9g}，下降 {(1-b['residual']/b['initial_residual'])*100:.3f}%。停止状态：{ {'iteration_budget_reached':'达到本轮迭代预算','residual_converged':'达到求根门限'}.get(b['status'],b['status']) }。求根门限仍为 1e−7。",convergence)
    def gates(ax):
        ax.semilogy(range(1,len(accepted)+1),[row['linear_gate'] for row in accepted],'o-',label='未受限 Newton 检查')
        ax.semilogy(range(1,len(accepted)+1),[max(row['trials'][row['accepted_trial']]['model_discrepancy'],1e-12) for row in accepted],'s-',label='受限步模型差异（h=1e−6）');ax.set(xlabel='接受步',ylabel='相对误差');ax.legend();ax.grid(alpha=.25)
    section('继续执行时的门限记录','实际步必须同时通过两个检查尺度、Cauchy 模型下界及完整映射下降检查。Newton 方向位于半径外时，不再单独阻断候选；位于半径内时仍保留 1% 检查。',gates)
if b:
    def rates(ax):
        values=np.array(vals);drops=100*(1-values[1:]/values[:-1]);ax.plot(np.arange(1,len(values)),drops,'o-');ax.set(xlabel='本轮接受步',ylabel='单步残差相对下降 (%)');ax.grid(alpha=.25)
    section('单步改善幅度与收敛速度','用相邻接受步的残差变化判断是否加速，不能仅依据残差单调下降宣称进入 Newton 快速区。',rates)
    def gradients(ax):
        for label in ['full_output_rebuild','cheap_current_residual']:
            rows=[(i+1,row) for i,row in enumerate(accepted) if row['gradient_source']==label]
            ax.semilogy([i for i,row in rows],[-row['checked_descent_slope'] for i,row in rows],'o',label={'full_output_rebuild':'新鲜完整输出受限方向','cheap_current_residual':'当前残差的廉价方向'}[label])
        ax.set(xlabel='本轮接受步',ylabel='经当前映射检查的单位方向下降斜率 −R·(Jd)');ax.legend();ax.grid(alpha=.25)
    section('新鲜受限梯度与廉价方向的下降强度','新鲜方向由完整输出残差计算，但状态方向仍限制在 C 子空间；它不是完整伴随梯度。廉价方向每步重新检查真实下降性。两者分别标记，不将投影子空间变平解释为全空间无根。',gradients)
if b and (ROOT/'endpoint_checks.json').exists():
    checks=json.loads((ROOT/'endpoint_checks.json').read_text());last5=float(np.mean((100*(1-np.array(vals[1:])/np.array(vals[:-1])))[-5:]))
    section('验证结果与剩余问题',f"30 个接受步均通过独立检查；其中 {checks['previously_blocked_but_accepted']} 步会被旧门限阻断。新鲜方向 {checks['fresh']} 次，廉价方向 {checks['cheap']} 次。末态重算差值 {checks['endpoint_difference']:.1g}。末 5 步平均仅改善 {last5:.5f}%，尚无快速收敛。末态场残差 {checks['residual_blocks']['field']:.6g}，反转残差 {checks['residual_blocks']['population']:.6g}；仍由场闭合控制。新鲜斜率存在波动，本轮不能证明全空间驻点。输出能量约 {checks['energy_nJ']:.6f} nJ，未达稳态认证。")
section('结论边界与可复核数据','本轮判断门限是否误挡可靠下降步，不把继续下降等同于找到周期一根，更不等同于稳定单脉冲。本轮未比较不同 Krylov 维数的受限模型，不能由内部 Newton 残差归零断言扩维对信赖域问题毫无作用。现有状态不构成 0.1–0.5 nJ 可用光源结果。需先闭合稳态残差，再做独立数值与动力学稳定性验证。')
links=''
for name in ['audit.json','continuation.json','endpoint_checks.json','protocol.json']:
    if (ROOT/name).exists():links+=f'<a download="{name}" href="data:application/json;base64,{base64.b64encode((ROOT/name).read_bytes()).decode()}">{name}</a> '
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：受限步门限验证</title><style>body{margin:auto;max-width:1200px;padding:24px;background:#edf2f7;color:#183044;font:18px/1.7 system-ui}section{background:white;padding:32px;margin:24px 0;border-radius:12px}h1,h2{color:#075681}img{width:100%;height:auto}a{overflow-wrap:anywhere}@media(max-width:600px){body{padding:10px}section{padding:15px}h1{font-size:25px}h2{font-size:21px}}</style><h1>原生10MHz：方向差分与受限步门限验证</h1>'+links+''.join(sections)+'</html>'
(ROOT/'原生10MHz_方向差分与受限步门限验证.html').write_text(html,encoding='utf8')
