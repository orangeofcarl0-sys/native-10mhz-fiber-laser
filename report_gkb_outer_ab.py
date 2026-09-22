"""Offline A/B evidence: progress, cost, radius probes and accepted guards."""
import base64,io,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':12})
a=json.loads((ROOT/'comparison.json').read_text(encoding='utf8'))
checks=json.loads((ROOT/'independent_checks.json').read_text(encoding='utf8'))
data={arm:json.loads((ROOT/arm/'continuation.json').read_text(encoding='utf8')) for arm in ['fresh','recycled']}
labels={'fresh':'A：fresh K192','recycled':'B：fresh K64 + old K64'};sections=[]
def table(head,rows):
    return '<div class="table"><table><tr>'+''.join('<th>'+str(x)+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'
def section(title,body,draw=None):
    pic=''
    if draw:
        fig,ax=plt.subplots(figsize=(11,5.8));draw(ax);fig.tight_layout();b=io.BytesIO();fig.savefig(b,format='png',dpi=150);fig.savefig(ROOT/f'figure_{len(sections)+1:02d}.png',dpi=150);plt.close(fig)
        pic='<img alt="'+title+'" src="data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()+'">'
    sections.append('<section id="s'+str(len(sections)+1)+'"><h2>'+title+'</h2>'+body+pic+'</section>')
def accepted(arm):return [r for r in data[arm]['history'] if 'accepted_trial' in r]
def trials(arm):return [r['trials'][r['accepted_trial']] for r in accepted(arm)]
def line(ax,values,ylabel):
    for arm in data:ax.plot(range(1,len(accepted(arm))+1),values(arm),'o-',label=labels[arm])
    ax.set(xlabel='接受步编号',ylabel=ylabel);ax.legend();ax.grid(alpha=.2)
section('本轮研究问题与冻结的A/B条件',
 '<p>从同一pure-GKB40末态R=9.746756121×10^-4出发，比较两个各20接受步的正式轨迹。A每状态重建fresh192；B构造fresh64，复用上一接受状态的fresh64方向，并用当前Jacobian重算其响应。B不携带旧JV、不携带联合128维基，也不使用旧gradient history。</p>'
 '<p>泵浦27.5 mW、OC=80%、GDD=+0.2 ps²、Psat=40 W、CNT→OC、20.42 m、10 MHz、网格与gauge全部不变。192维是固定的统一基线，不是预先认定的最优深度。两臂共用初始半径0.0125及相同controller。顺序使用同一GPU；运行期间不并行其他数值验证。</p>'
 '<p>B首步old64来自已保存的pure40上一个状态（states[-2]）。这是warm-cache续算，不是凭空免费的全新启动；下文另列旧缓存重建成本参考。</p>')
section('固定空间内如何选择有限步长',
 '<p>每状态只构建一次空间。先试上一接受半径；实际下降非正、ρ&lt;0.25或基本保护失败时，半径减半重解。ρ&gt;0.8且边界解时，额外试1.5倍半径；新候选实际下降更大且ρ&gt;0.5才继续外试。</p>'
 '<p>最终按真实ΔΦ从高到低检查所有已试且基本合格的候选，选择通过三尺度Js检查者。保留可行性、正实际与预测、ρ&gt;0.1、同半径完整梯度Cauchy下界、步范数约束；三尺度h=10^-5/3×10^-6/10^-6，预测一致性5%。εNL只诊断，不门限。</p>'
 '<p>因此探索停止规则和最终接受规则不同：某个外试点即使ρ未超过0.5，只要它真实下降最大且通过全部接受保护，仍可成为最终候选。算法不是选择ρ最接近1的点。</p>')
section('两条轨迹的总结果与预设决策门',table(['指标','A：fresh192','B：recycled'],[
 ['接受步数',*[data[arm]['accepted_steps'] for arm in data]],
 ['最终R',*[f'{data[arm]["residual"]:.10g}' for arm in data]],
 ['P=-log(Rend/R0)',*[f'{data[arm]["progress"]:.6g}' for arm in data]],
 ['算法时间/s',*[f'{data[arm]["algorithm_seconds"]:.2f}' for arm in data]],
 ['basis时间/s',*[f'{data[arm]["basis_seconds"]:.2f}' for arm in data]],
 ['nonlinear候选数',*[data[arm]['nonlinear_probes'] for arm in data]],
 ['结束状态',*[data[arm]['status'] for arm in data]]])+
 f'<p>P_B/P_A=<b>{a["progress_ratio"]:.4f}</b>；T_B/T_A=<b>{a["time_ratio"]:.4f}</b>；basis时间比={a["basis_time_ratio"]:.4f}。预设门为推进比≥0.9且总算法时间比≤0.75，并要求完成20步或提前达根。本次门：<b>{"通过" if a["supports_recycled_default"] else "未通过"}</b>。</p>')
def residual_plot(ax):
    for arm in data:
        norms=[data[arm]['initial_residual']]+[t['residual'] for t in trials(arm)]
        ax.semilogy(range(len(norms)),norms,'o-',label=labels[arm])
    ax.set(xlabel='接受步编号',ylabel='完整残差范数R');ax.legend();ax.grid(alpha=.2)
section('随接受步数的闭合残差变化',
 '<p>每个点来自实际接受状态，而非冻结审计的候选。相同步数比较收敛推进，不代表相同计算预算；下一图补充时间口径。曲线下降不等于已找到1e-7周期根，也不等于稳定单脉冲认证。</p>',residual_plot)
def time_plot(ax):
    for arm in data:
        times=np.r_[0,np.cumsum([r['seconds'] for r in accepted(arm)])]
        norms=np.array([data[arm]['initial_residual']]+[t['residual'] for t in trials(arm)])
        ax.plot(times,-np.log(norms/norms[0]),'o-',label=labels[arm])
    ax.set(xlabel='累计算法时间/s（不含文件归档）',ylabel='累计推进P');ax.legend();ax.grid(alpha=.2)
section('相同计算时间口径下的累计推进',
 '<p>P使用自然对数，表示残差的乘法式下降。横轴包含每步构基、响应重算、合并、半径候选、最终导数门及必要数值工作，不包含observer压缩归档。文件写入的实耗另外保存在wall_seconds。</p>',time_plot)
section('每个接受步的真实目标下降',
 '<p>纵轴ΔΦ=½(Rbefore²−Rafter²)。B每步所用old64都来自其自身轨迹的上一接受状态，不借用A当前或未来信息。两条轨迹走向不同状态后，差异是整套更新策略的累计效果。</p>',lambda ax:line(ax,lambda arm:[t['actual'] for t in trials(arm)],'真实目标下降ΔΦ'))
section('接受半径与各步候选数量',table(['步','A半径','A probes','B半径','B probes'],[[i+1,*sum(([f'{trials(arm)[i]["radius"]:.6g}',len(accepted(arm)[i]['trials'])] for arm in data),[])] for i in range(min(len(accepted(arm)) for arm in data))]),
 lambda ax:line(ax,lambda arm:[t['radius'] for t in trials(arm)],'最终选中半径Δ'))
section('接受半径与实际步范数的区别','<p>相同半径并不保证相同实际步长；如果λ=0，空间内的无约束步可能远小于半径。该图用于区分半径限制与当前空间给出的步本来较小。每个候选的λ与boundary标记均保存在原始记录。</p>',lambda ax:line(ax,lambda arm:[t['step_norm'] for t in trials(arm)],'实际步范数'))
section('最终候选的ρ与真实性验收',
 '<p>选中候选全部满足ρ&gt;0.1；实际下降更大的候选优先，而不是人为强制ρ≈1。各半径的失败与未选候选保留在JSON及完整逐步表。</p>',lambda ax:line(ax,lambda arm:[t['rho'] for t in trials(arm)],'接受候选ρ'))
section('接受步的有限步长非线性缺陷',
 '<p>εNL=‖R(x+s)−R(x)−Js‖/‖Js‖。它测残差向量非线性，与merit比ρ不是同一个量。数值较大不会直接拒绝；是否继续下降由真实腔映射和保护判断。</p>',lambda ax:line(ax,lambda arm:[t['nonlinear_defect'] for t in trials(arm)],'非线性缺陷εNL'))
def cost(ax):
    for arm in data:ax.plot(range(1,len(accepted(arm))+1),[r['basis_seconds'] for r in accepted(arm)],'o-',label=labels[arm]+' 构基')
    ax.set(xlabel='接受步编号',ylabel='构基/当前响应/合并总时间s');ax.legend();ax.grid(alpha=.2)
section('Jacobian信息成本与半径探索成本',
 table(['时间项/s','A','B'],[[name,*[f'{data[arm][name]:.2f}' for arm in data]] for name in ['basis_seconds','probe_seconds','gate_seconds','algorithm_seconds','wall_seconds']])+
 '<p>probe_seconds包括小SVD、候选形成和真实残差评估；gate_seconds为最终候选的三尺度检查。basis包含B对旧64方向的当前响应重算，不把这些Jv算作免费。</p>',cost)
def blocks(ax):
    for arm in data:
        for block,style in [('field','-'),('population','--'),('gauge',':')]:
            ax.semilogy(range(1,len(accepted(arm))+1),[r['blocks'][block] for r in accepted(arm)],style,label=labels[arm]+' '+block)
    ax.set(xlabel='接受步编号',ylabel='残差块范数');ax.legend(fontsize=9);ax.grid(alpha=.2)
section('光场闭合、反转与gauge的分项残差',
 '<p>分块用于判断剩余闭合误差来自哪里；field是两偏振复光场闭合，population是EDF反转，gauge是固定相位与平移约束。该分解尚不能单独归因于某个光学器件。</p>',blocks)
section('缓存初始化、数值验证与复现边界',
 f'<p>B首步使用已经保存的旧64基。上一轮独立重建参考成本为{a["cold_cache_rebuild_reference_seconds"]:.2f}s；将其加到B总时间后，参考cold比为{(data["recycled"]["algorithm_seconds"]+a["cold_cache_rebuild_reference_seconds"])/data["fresh"]["algorithm_seconds"]:.4f}。它是历史机器计时参考，不是假装本轮重新测量。</p>'+
 table(['臂','独立复放接受步','三尺度预测最大相对误差'],[[labels[arm],checks[arm]['accepted_replayed'],f'{checks[arm]["maximum_prediction_relative_error"]:.4g}'] for arm in data])+
 '<p>已知答案与控制器测试通过；独立复核同起点、每步完整残差、三尺度方向导数、Cauchy下界、步长及已试候选的最大真实下降选择。两臂是确定性单次轨迹，没有用统计置信区间包装机器计时；热状态或系统负载仍可能影响秒数。</p>')
section('子空间数值健康与回收秩',table(['指标','A','B'],[[key,*[f'{max(row[key] for row in data[arm]["history"]):.4g}' for arm in data]] for key in ['u_orth','v_orth','bidiagonal_relative']])+ '<p>联合空间采用rank-aware QR/SVD正交化，按奇异值最大值的10^-12截断；同步变换J响应，保持既有状态欧氏信赖域度量。上述量为全轨迹最大误差；每步实际秩、奇异谱保存在JSON。</p>')
section('完整逐步候选记录', '<details><summary>展开所有半径候选及选择结果</summary>'+table(['臂','步','Δ','Δm','ΔΦ','ρ','选中','最终gate'],[[arm,row['step'],f'{t["radius"]:.6g}',f'{t["prediction"]:.6g}',f'{t["actual"]:.6g}' if t['actual'] is not None else '不可行',f'{t["rho"]:.4f}',i==row.get('accepted_trial'),('通过' if t['passed'] else '未通过') if t['checks'] else '未执行（未选）'] for arm in data for row in data[arm]['history'] for i,t in enumerate(row['trials'])])+'</details>')
section('后段推进、无约束步与空间信息的关系',table(['后段指标','A','B'],[['末5步平均相对下降',*[f'{100*np.mean([1-trials(arm)[i]["residual"]/trials(arm)[i-1]["residual"] for i in range(15,20)]):.4f}%' for arm in data]],['边界解接受步数',*[sum(t['boundary'] for t in trials(arm)) for arm in data]]])+ '<p>B第2–20步均为λ=0的空间内无约束解；扩大半径不会改变这些reduced LS解。B末5步范数约0.00132–0.00171，A约0.00563–0.00595；两臂ρ均通过验收。这与回收空间产生的有效最小二乘步较小相符，不能归咎于把B的信赖半径压得过小。两条轨迹已处于不同状态，因此这不是同状态的唯一机制证明。</p>')
section('本轮证据支持的算法选择',
 ('<p><b>预设门通过：在当前物理设置与20步续算区间，支持将回收GKB作为生产默认候选。</b>这不等于所有状态都占优；仍需保留完整半径搜索和最终方向检查。</p>' if a['supports_recycled_default'] else f'<p><b>预设门未通过：保留fresh K192＋内层半径控制作为当前生产基线，不将纯K64+old64切为默认。</b>B虽只需{100*a["time_ratio"]:.2f}%的时间，但也只有{100*a["progress_ratio"]:.2f}%的累计推进；按P/T计算，B约为A的{100*a["progress_ratio"]/a["time_ratio"]:.1f}%，没有显示平均推进效率提升。该平均量的时间区间不同，应同时看完整时间曲线。</p><p>冻结单点中回收空间的优势未持续转化为跨20步的同等推进。hybrid仍是待验证方案，本轮不据此制定refresh触发器，更不转multiple shooting。</p>')+
 '<p>本轮未设计hybrid refresh触发器、未进行自适应深度、未引入预条件或multiple shooting，也未改写旧入口。数值根、动力学稳定性与实验可实现性是后续不同层次的验证，不能由本A/B直接替代。</p>')
style='body{margin:0;background:#edf2f6;color:#182d40;font:17px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:25px}section{background:white;border-top:4px solid #176c91;padding:28px;margin:25px 0}h1{font-size:30px}h2{font-size:24px;margin-top:0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cddbe4;padding:8px;text-align:left}th{background:#e8f1f6}.table{overflow:auto}a{color:#176c91}@media(max-width:600px){main{padding:10px}section{padding:14px}h1{font-size:24px}h2{font-size:20px}}'
payload=base64.b64encode(json.dumps(dict(comparison=a,trajectories=data,checks=checks),ensure_ascii=False).encode()).decode()
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：深层与回收GKB正式A/B</title><style>'+style+'</style><main><h1>原生10MHz：深层／回收GKB与内层半径控制20步对照</h1>'+''.join(sections)+f'<a download="outer_ab.json" href="data:application/json;base64,{payload}">下载完整记录</a></main></html>'
(ROOT/'原生10MHz_深层与回收GKB正式20步对照.html').write_text(html,encoding='utf8');print('REPORT',len(sections))
