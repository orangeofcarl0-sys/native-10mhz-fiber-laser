"""Offline scientific report of the adaptive pure GKB trajectory."""
import base64,io,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':12})
a=json.loads((ROOT/'continuation.json').read_text(encoding='utf8'))
c=json.loads((ROOT/'independent_checks.json').read_text(encoding='utf8'))
rows=[r for r in a['history'] if 'accepted_trial' in r]
trials=[r['trials'][r['accepted_trial']] for r in rows];x=np.arange(1,len(rows)+1)
sections=[]
def table(head,values):
    return '<div class="table"><table><tr>'+''.join('<th>'+v+'</th>' for v in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in values)+'</table></div>'
def section(title,body,draw=None):
    picture=''
    if draw:
        fig,ax=plt.subplots(figsize=(11,5.5));draw(ax);fig.tight_layout();b=io.BytesIO()
        fig.savefig(b,format='png',dpi=150);fig.savefig(ROOT/f'figure_{len(sections)+1:02d}.png',dpi=150);plt.close(fig)
        picture='<img alt="'+title+'" src="data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()+'">'
    sections.append('<section id="s'+str(len(sections)+1)+'"><h2>'+title+'</h2>'+body+picture+'</section>')
section('固定物理模型下的纯GKB正式轨迹',
 '<p>问题是：用最小二乘 GKB 取代原 root-GMRES 主空间后，period-1 单段闭合残差能否跨越原来的慢平台？本轮使用同一完整腔映射与离散伴随，固定泵浦 27.5 mW、OC=80%、GDD=+0.2 ps²、Psat=40 W、CNT→OC、20.42 m、10 MHz、网格和规范模板。</p>'
 f'<p>从伴随40步的末态 R={a["initial_residual"]:.10g} 开始，初始 Δ=0.00625；没有采用冻结审计的试探终点。本轮实际接受 <b>{a["accepted_steps"]} 步</b>，停止原因 {a["status"]}，最终 <b>R={a["residual"]:.10g}</b>，相对起点下降 {100*(1-a["residual"]/a["initial_residual"]):.3f}%。</p>')
section('每步如何扩展最小二乘空间',
 '<p>u₁=−R/‖R‖，交替调用 Jᵀu 和 Jv，对 U、V 做两遍全重正交，得到 JV≈UB。状态步 s=Vy；在当前缩放状态欧氏度量中，‖s‖=‖y‖，因此直接求 min ½‖βe₁−By‖²，约束 ‖y‖≤Δ。</p>'
 '<p>每 8 维检查一次最优模型下降，连续两次 [Δmₖ−Δmₖ₋₈]/Δmₖ&lt;1% 才停，否则扩到64。小 B 用 SVD 和标量乘子求根，不形成完整 JᵀJ。缩半径重试复用同一个状态的已生成空间；接受后才换状态重新生成。</p>'
 '<p>本轮无 history、root-Z、C 预条件器、annulus 或 seed。k=1 自然提供完整梯度 Cauchy 基准；运行时仍检查候选不劣于它。方法参考 <a href="https://web.stanford.edu/group/SOL/software/lsqr/">Stanford SOL 的 Golub–Kahan / LSQR 说明</a>；主实现 steady_gkb.py、steady_gkb_outer.py。</p>')
def normplot(ax):
    ax.plot(np.arange(len(a['norms'])),a['norms'],'o-',ms=3,label='正式轨迹')
    for t in [1e-3,1e-4,1e-5]:
        if min(a['norms'])*.8<=t<=max(a['norms'])*1.2:ax.axhline(t,ls='--',alpha=.5,label=f'R={t:g}')
    ax.set(xlabel='接受步数',ylabel='完整残差范数');ax.legend();ax.grid(alpha=.2)
section('完整残差与数量级里程碑',
 '<p>横轴是数值求解器的接受步，不是光在腔内的物理往返时间。所有点来自一条连续轨迹，不能将它解释为锁模建立时间。</p>'+
 table(['阈值','首次达到的接受步'],[[k,v if v is not None else '未达到'] for k,v in a['milestones'].items()]),normplot)
def gains(ax):
    g=100*(1-np.array(a['norms'][1:])/np.array(a['norms'][:-1]));ax.plot(x,g,'o-',ms=3,label='单步')
    if len(g)>=5:ax.plot(x[4:],np.convolve(g,np.ones(5)/5,'valid'),label='末5步平均')
    ax.set(xlabel='接受步数',ylabel='残差相对下降 / %');ax.legend();ax.grid(alpha=.2)
section('多步推进速度',f'<p>末五步平均相对残差下降为 {100*a["last_five_mean_relative_decrease"]:.5f}%/步。单步真实 merit 下降与残差范数下降是不同量，此图使用后者。</p>',gains)
def dimension(ax):
    ax.step(x,[r['k'] for r in rows],where='mid',label='使用维数')
    sat=[i for i,r in enumerate(rows) if r['saturated']]
    if sat:ax.scatter(x[sat],[rows[i]['k'] for i in sat],marker='s',label='满足连续两次<1%')
    ax.set(xlabel='接受步数',ylabel='GKB 维数',ylim=(0,68));ax.legend();ax.grid(alpha=.2)
section('自适应维数与停维原因',
 f'<p>{sum(r["saturated"] for r in rows)} 个接受步满足饱和规则。达到64维上限本身不代表收益已经饱和，必须看边际预测增益。</p>',dimension)
def marginal(ax):
    values=[100*r['ladder'][-1].get('marginal_gain',np.nan) for r in rows]
    ax.plot(x,values,'o-',ms=3);ax.axhline(1,ls='--',color='red',label='1%参考线')
    ax.set(xlabel='接受步数',ylabel='最后增加8维的相对模型收益 / %');ax.legend();ax.grid(alpha=.2)
section('达到维数上限后是否仍有扩维收益',
 '<p>纵轴为最后一个检查节点相对前8维的收益 [Δmₖ−Δmₖ₋₈]/Δmₖ，均在本步初始半径计算。若持续显著高于1%，即使真实下降已经变慢，也不能说最小二乘空间充分展开。</p>',marginal)
def ladder(ax):
    for i in sorted(set([0,len(rows)//2,len(rows)-1])):
        rr=[r for r in rows[i]['ladder'] if 'prediction' in r]
        ax.plot([r['k'] for r in rr],[r['prediction']/rr[-1]['prediction'] for r in rr],'o-',label=f'步{i+1}, Δ={rows[i]["radius"]:.4g}')
    ax.set(xlabel='GKB 维数',ylabel='模型下降 / 本步最终维数模型下降');ax.legend();ax.grid(alpha=.2)
section('代表状态的扩维收益',
 '<p>取起步、中点和末步，不按效果挑选状态。曲线在各自 outer 初始半径计算；若候选随后缩半径，接受半径另外记录。完整所有步、所有检查节点保存在可下载JSON。</p>',ladder)
def spectrum(ax):
    for i in sorted(set([0,len(rows)//2,len(rows)-1])):
        values=rows[i]['singular_values'];ax.semilogy(range(1,len(values)+1),values,label=f'步{i+1}')
    ax.set(xlabel='小矩阵B奇异值序号',ylabel='奇异值');ax.legend();ax.grid(alpha=.2)
section('最小二乘小矩阵的奇异值谱',
 '<p>这里是已生成 Krylov 空间的小矩阵 B 的谱，不是完整腔 Jacobian 的全谱。小奇异值表示该子空间中响应弱的状态组合，不能仅凭它断言分岔或真实物理不稳定。</p>',spectrum)
def blocks(ax):
    for key in ['field','population','gauge']:
        ax.semilogy(range(len(a['blocks'])),[max(r[key],1e-30) for r in a['blocks']],label=key)
    ax.set(xlabel='接受步数',ylabel='残差块范数');ax.legend();ax.grid(alpha=.2)
section('场闭合、反转与规范条件的残差',
 '<p>field 对应一圈后光场相对初态的闭合，population 对应增益反转自洽，gauge 固定相位和时移自由度。三项均沿用原缩放，各自变化用于定位哪一部分仍未闭合。</p>',blocks)
def rho(ax):
    ax.plot(x,[t['rho'] for t in trials],'o-',ms=3);ax.axhline(.1,ls='--',color='red');ax.set(xlabel='接受步数',ylabel='真实下降 / 模型下降（ρ）');ax.grid(alpha=.2)
section('信赖模型与真实腔映射的一致性',
 f'<p>保留全部拒绝尝试，接受 {len(trials)} 个候选，拒绝 {sum(len(r["trials"]) for r in a["history"])-len(trials)} 个。图中仅画接受步ρ。三尺度 Js、Cauchy 下界、状态可行性和正实际下降均为必要条件；fallback 使用时另行标注。</p>',rho)
section('数值复核、正交性与运行成本',
 f'<p>独立复算 {c["states_replayed"]} 个状态及 {c["steps_verified"]} 个接受步，三尺度预测最大相对误差 {c["maximum_prediction_error"]:.4g}。'
 f'U/V 最大正交误差分别 {max(r["u_orth"] for r in rows):.3g} / {max(r["v_orth"] for r in rows):.3g}，JV−UB 最大相对误差 {max(r["bidiagonal_relative"] for r in rows):.3g}。</p>'
 f'<p>总墙钟 {a["seconds"]:.1f} s；每步GKB构造中位数 {np.median([r["gkb_seconds"] for r in rows]):.2f} s。运行耗时包含重正交与伴随原映射重放，不能仅按维数线性外推。</p>'+
 table(['步','R','k','接受Δ','ρ','步范数','类型'],[[i+1,f'{t["residual"]:.7g}',r['k'],f'{t["radius"]:.5g}',f'{t["rho"]:.4f}',f'{t["step_norm"]:.5g}',t['kind']] for i,(r,t) in enumerate(zip(rows,trials))]))
section('本轮结果能支持的判断',
 f'<p>数值根阈值 R&lt;10⁻⁷：{"达到" if a["numerical_root"] else "未达到"}。预设 multiple-shooting 诊断门：{"满足" if a["multiple_shooting_diagnostic"] else "未满足"}。</p>'
 f'<p>本轮总残差下降 {100*(1-a["residual"]/a["initial_residual"]):.3f}%，首次跨过10⁻³在第 {a["milestones"]["0.001"]} 步。末五步平均下降 {100*a["last_five_mean_relative_decrease"]:.5f}%/步，尚无数量级跃迁。末态场项占残差平方范数的 {100*(a["blocks"][-1]["field"]/a["residual"])**2:.5f}%，仍主要是场闭合问题。</p>'
 f'<p>末五步最后8维的模型收益范围为 {100*min(r["ladder"][-1].get("marginal_gain",0) for r in rows[-5:]):.2f}%–{100*max(r["ladder"][-1].get("marginal_gain",0) for r in rows[-5:]):.2f}%。因此慢推进与“空间已饱和”不能画等号；本次64维上限试验尚不能排除有限 Krylov 预算的影响，也不能证明 single-shooting formulation 本身无法闭合。</p>'
 '<p>该诊断要求至少30个接受步，末10步均在48–64维且满足停维饱和、ρ中位数&gt;0.5，末5平均相对下降&lt;0.05%且R≥10⁻⁴。若64维仍未饱和，就不能说搜索空间已经充分展开。本轮不会自动切换 formulation。</p>'
 '<p>残差收敛只是数值闭合证据；稳定单脉冲还需要网格、窗口、扰动与物理稳定性等独立验证。本轮不以轨迹下降替代这些认证。</p>')
payload=base64.b64encode(json.dumps(dict(result=a,checks=c),ensure_ascii=False).encode()).decode()
style='body{margin:0;background:#edf2f6;color:#182d40;font:17px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:25px}section{background:white;border-top:4px solid #176c91;padding:28px;margin:25px 0}h1{font-size:30px}h2{font-size:24px;margin-top:0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cddbe4;padding:8px;text-align:left}th{background:#e8f1f6}.table{overflow:auto}a{color:#176c91}@media(max-width:600px){main{padding:10px}section{padding:14px}h1{font-size:24px}h2{font-size:20px}}'
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：纯GKB正式续算</title><style>'+style+'</style><main><h1>原生10MHz：纯GKB自适应64维正式续算</h1>'+''.join(sections)+f'<p><a download="continuation.json" href="data:application/json;base64,{payload}">下载完整轨迹记录</a></p></main></html>'
(ROOT/'原生10MHz_纯GKB自适应64维正式续算.html').write_text(html,encoding='utf8')
print('REPORT',len(sections))
