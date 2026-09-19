"""Chinese offline report for the paired nonlinear trajectory."""
import base64,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':12})
runs=[json.loads((ROOT/(arm+'.json')).read_text()) for arm in ['A','B']]
assert runs[0]['protocol']==runs[1]['protocol'], 'Pair protocol mismatch'
assert runs[0]['protocol']['pump_W']==.0275, 'This report covers the 27.5 mW experiment'
for required in ['endpoint_checks.json','linear_budget_check.json']:
    if not (ROOT/required).exists():raise FileNotFoundError('Run both check_augmented_pair.py and check_augmented_linear_budget.py before reporting')

def accepted(run):return [h for h in run['history'] if 'accepted_trial' in h]
def selected(h):return h['trials'][h['accepted_trial']]
def trajectory(run):return np.array([run['initial_residual']]+[selected(h)['residual'] for h in accepted(run)])
def table(head,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+v+'</th>' for v in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
def fig(name,draw):
    f,ax=plt.subplots(figsize=(11,5));draw(ax);ax.grid(alpha=.2);p=ROOT/(name+'.png');f.savefig(p,dpi=160,bbox_inches='tight');plt.close(f)
    return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
def curves(ax,fun,ylabel,title,log=False):
    for r in runs:
        vals=fun(r);ax.plot(np.arange(len(vals)),vals,'o-',ms=3,label=r['arm']+('：原Hookstep' if r['arm']=='A' else '：增广Hookstep'))
    ax.set(xlabel='接受步数 / 从0开始',ylabel=ylabel,title=title)
    if log:ax.set_yscale('log')
    ax.legend()

sections=[]
sections.append('<h2>1．研究问题与配对实验条件</h2><p>上一轮已证明：特定状态下加入经验证的下降方向能够改善单步结果。本轮检验这种改善能否在更新状态后持续，并进入更快的周期一求根阶段。A保留原GMRES–Hookstep；B增加下降方向、增广子空间和Cauchy回退。</p><p>两组从完全相同的down_275出发，固定泵浦27.5 mW、腔长20.42 m、CNT → OC、OC 80%、净色散+0.2 ps²。窗口2.048 ns、dt=0.125 ps；残差/状态缩放和规范模板不变，输出能量不进入目标函数。预算均为40个outer steps；不是重新扫描泵浦。</p>'+table(['共用设置','数值'],[['C预条件器','前向差分1e-6；事件触发重建'],['Arnoldi','中心Jv 1e-5；240维；目标相对残差0.008'],['独立线性门限','中心差分1e-6；无约束Newton方向相对线性残差 < 0.01'],['信赖域','初始0.003125，上限2；ρ>0.75且到达边界时翻倍，ρ<0.25缩为1/4'],['接受条件','预测及独立模型的ρ均>0.1，实际merit下降且状态可行'],['根阈值','完整残差范数 < 1e-7；求根不等于动力学稳定']]))
sections.append('<h2>2．下降方向更新与三层保护</h2><p>重建时，令E为粗空间提升，响应列为JE；直接累计g=(JE)ᵀR，再取d=−Eg/‖Eg‖。累计发生在输出投影之前，不额外调用腔映射。中间步使用已存A=PJ_oldE和当前R生成g=AᵀPR。这个廉价方向不是当前精确梯度，必须用当前完整映射的中心Jd，以及独立差分尺度再检查RᵀJd&lt;0。</p><p>模型层：原始增广步预测下降不得小于Cauchy的(1−1e-6)，否则显式记录失败并回退。实际层：用完整非线性映射计算ρ，不合格就缩半径重试，末次仍失败则检查Cauchy实际步。刷新层：线性超限或方向不下降立即在当前状态重建；低ρ或模型预测差异>5%标记下一步重建。A/B共用适用的门限，B另有下降方向检查。</p><p>G_C与G_H均为所选步的模型下降除以同一状态、同一半径下Cauchy/原Hookstep的模型下降，不能解释为物理精度倍数。投影覆盖率χ是长度占比；误差ε=‖d−P_Zd‖/‖d‖。保存原始增广验收与回退标志，避免把回退算作增广成功。</p>')
status_label={'linear_accuracy_limited':'独立线性检查超限','iteration_budget_reached':'达到迭代预算','residual_converged':'达到根阈值'}
summary=table(['组别','已接受步数 / 预算','停止原因','初始残差','最终残差','累计下降','耗时/s','重建次数'],[[r['arm'],f"{r['accepted_steps']} / 40",status_label.get(r['status'],r['status']),f"{r['initial_residual']:.8g}",f"{r['physical_residual']:.8g}",f"{100*(1-r['physical_residual']/r['initial_residual']):.4f}%",f"{r['seconds']:.1f}",sum(len(h['builds']) for h in r['history'])] for r in runs])
common=min(r['accepted_steps'] for r in runs);ratio=float(trajectory(runs[1])[common]/trajectory(runs[0])[common]);replicate=(common>=30 or runs[1]["numerical_root"]) and ratio<=.8
text='<h2>3．累计残差与可比较的迭代区间</h2>'+summary
text+=f'<p>在共同完成的第{common}个接受步，B/A残差比为{ratio:.6g}。预先约定的27.0 mW复核触发条件为：至少完成30个共同接受步（或B提前达到根阈值），且B残差比A低至少20%；本轮触发：{"是" if replicate else "否"}。这是确定性效果大小标准，不是统计显著性检验。</p>'
def convergence(ax):
    curves(ax,trajectory,'完整残差范数','A/B从同一状态出发的残差轨迹',True)
    if runs[0]['accepted_steps']!=runs[1]['accepted_steps']:
        ax.axvline(common,color='gray',ls='--',label='共同步数区间结束');ax.legend()
text+=table(['共同接受步数','A残差','B残差','A累计下降','B累计下降'],[[common,f"{trajectory(runs[0])[common]:.9g}",f"{trajectory(runs[1])[common]:.9g}",f"{100*(1-trajectory(runs[0])[common]/runs[0]['initial_residual']):.5g}%",f"{100*(1-trajectory(runs[1])[common]/runs[1]['initial_residual']):.5g}%"]])
text+=fig('累计残差',convergence)
text+=fig('单步下降',lambda ax:curves(ax,lambda r:100*(1-trajectory(r)[1:]/trajectory(r)[:-1]),'实际残差下降/%','逐步改善是否持续'))
sections.append(text)
text='<h2>4．信赖半径与非线性接受比</h2>'
text+=fig('信赖半径',lambda ax:curves(ax,lambda r:[selected(h)['radius'] for h in accepted(r)],'接受步对应的半径Δ','自适应半径轨迹',True))
text+=fig('实际接受比',lambda ax:curves(ax,lambda r:[selected(h)['rho'] for h in accepted(r)],'实际merit下降 / 模型下降','接受步的ρ'))
text+='<p>半径是否扩张由实际ρ和边界步决定，未把旧子空间的半径扫描结果当成新增广方法的固定上限。图中只画接受步；所有失败试探与低ρ事件均保存在下载的历史JSON中。</p>'
sections.append(text)
b=accepted(runs[1]);text='<h2>5．增广收益、方向覆盖与步长</h2>'
def gain(ax):
    for k in ['G_C','G_H']:ax.plot([h['step'] for h in b],[selected(h).get(k,np.nan) for h in b],'o-',label=k)
    ax.set(xlabel='B组外层迭代编号',ylabel='模型下降比',title='所选步相对Cauchy及原Hookstep的收益');ax.legend()
text+=fig('模型收益',gain)
text+='<p>比值较大也可能来自分母较小，需与前面的实际单步下降图一起看，不能把G_H=10解释成残差本身缩小10倍。</p>'
def coverage(ax):
    ax.plot([h['step'] for h in b],[100*h['span'][0]['coverage'] for h in b],'o-')
    ax.set(xlabel='B组外层迭代编号',ylabel='方向投影长度占比χ/%',title='当前下降方向在原Krylov空间中的覆盖')
text+=fig('方向覆盖',coverage)
def norms(ax):
    for k,label in [('newton_norm','无约束Newton步'),('step_norm','实际接受步'),('alpha_cauchy','Cauchy步长')]:
        ax.semilogy([h['step'] for h in b],[h[k] if k in h else selected(h).get(k,np.nan) for h in b],'o-',label=label)
    ax.set(xlabel='B组外层迭代编号',ylabel='缩放状态范数 / D=I',title='方向可用性与步长尺度');ax.legend()
text+=fig('步长尺度',norms)
text+=table(['B步','方向来源','ε','α_C','G_C','G_H','接受类型'],[[h['step'],h['gradient_source'],f"{h['span'][0]['epsilon']:.6g}",f"{selected(h).get('alpha_cauchy',float('nan')):.5g}",f"{selected(h).get('G_C',float('nan')):.5g}",str(selected(h).get('G_H')),h['selected_kind']] for h in b])
sections.append(text)
text='<h2>6．线性门限与重建事件</h2>'
def gate(ax):
    for r in runs:ax.semilogy([h['step'] for h in r['history']],[h['linear_gate'] for h in r['history']],'o-',label=r['arm']+('：原Hookstep' if r['arm']=='A' else '：增广Hookstep'))
    ax.axhline(.01,color='r',ls='--',label='1%门限');ax.set(xlabel='外层迭代编号',ylabel='独立Newton线性相对残差',title='当前状态的独立线性精度');ax.legend()
text+=fig('独立线性门限',gate)
text+=fig('两组Newton步',lambda ax:curves(ax,lambda r:[h['newton_norm'] for h in r['history']],'无约束Newton步范数','两组求根方向的步长尺度',True))
text+=table(['组','步','重建原因','线性检查'],[[r['arm'],h['step'],', '.join(v['reasons']),f"{h['linear_gate']:.6g}"] for r in runs for h in r['history'] for v in h['builds']])
text+=table(['组','停止步','Arnoldi维数','内部预测','独立检查'],[[r['arm'],h['step'],v['dimension'],f"{v['model']:.6g}",f"{v['independent']:.6g}"] for r in runs for h in r['history'] if 'accepted_trial' not in h for v in h['linear_attempts']])
if (ROOT/'linear_budget_check.json').exists():
    budget=json.loads((ROOT/'linear_budget_check.json').read_text())
    text+=table(['停止点复核','强制维数','内部预测','独立检查','通过1%门限'],[[v['arm'],v['dimension'],f"{v['model']:.6g}",f"{v['independent']:.6g}",str(v['gate_pass'])] for v in budget])
text+='<p>如果新建预条件器后仍不能满足1%门限，则不接受该次非线性步，停止原因记为linear_accuracy_limited。这表明当前线性求解预算/精度限制了本轮对照，不能视为非线性方程无根。补充强制240维后，A/B内部模型残差分别降至4.09e-11 / 7.68e-12，但独立差分检查仍为3.009% / 1.861%，并未通过门限；因此两组终点停止都不能仅归因于Arnoldi内部提前停止。</p>'
sections.append(text)
text='<h2>7．结论范围与后续判据</h2><p id="result-interpretation">两组同样完成18个接受步时，B减少的残差量约为A的3.45倍，支持单步优势转化为累计改善。A最终接受18步，B接受30步；两组均因独立线性门限停止，没有完成40步预算。B累计下降5.946%，末5步平均每步只下降约0.0424%，没有观察到进入Newton快速收敛区。全部30个B接受步均为原始增广步：9次使用重建得到的完整输出受限梯度，21次使用通过当前状态检查的廉价方向。</p><p>残差是缩放后周期一联合方程的不闭合量，不是测距精度，也不是输出功率误差。单步/累计下降只能说明数值求解推进；只有残差小于1e-7才进入周期一根检查，之后还需窗口、离散误差和扰动稳定性验证。本轮诊断态的输出能量仍约0.9 nJ，不是已达到0.1–0.5 nJ目标的可用光源。没有得到根不能直接推出不存在period-1，更不能据此认定必须period-p。下一步应先检查这些大范数Newton方向的Jv差分尺度及线性组合一致性，区分差分误差与条件数放大的影响。</p>'
if (ROOT/'endpoint_checks.json').exists():
    checks=json.loads((ROOT/'endpoint_checks.json').read_text())
    text+=table(['状态','光场子残差','反转子残差','规范子残差'],[['初始']+[f"{checks['A']['initial_blocks'][k]:.7g}" for k in ['field','population','gauge']]]+[[arm]+[f"{checks[arm]['residual_blocks'][k]:.7g}" for k in ['field','population','gauge']] for arm in ['A','B']])
    text+='<p>以上是已经按模型固定尺度归一化后的子残差范数，其平方和等于完整残差的平方。终点由独立脚本重新计算，并检查接受历史连续性、单调下降、独立ρ和线性门限。B终点的反转子残差从1.41e-4降至1.32e-5，规范子残差从3.42e-7升至5.68e-6；剩余merit约99.99%来自光场块。因此可以说总merit与光场/反转块改善，不能声称每个子方程都同步改善。</p>'
sections.append(text)
sections.append('<h2>8．复现、数据与实现核验</h2><p>steady_support_lu.py新增可选descent缓存，未改变原默认求逆。steady_augmented_outer.py提供共用外循环；run_augmented_pair.py运行A/B；check_augmented_pair.py复核终点；check_augmented_linear_budget.py补充满预算检查；最后由build_augmented_pair_report.py生成本报告。需配置CuPy并设置独立LASER_OUTPUT_DIR。公开结果含每次试探、刷新原因、终态及执行源码快照；输入SHA与代码SHA写入结果，旧报告保留。</p><p>小型已知答案测试验证梯度累计不增加映射次数、更新状态后的廉价方向可用、两组均收敛到线性方程真根、半径可越过0.00625；原增广测试覆盖强制失败回退。</p>')
body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：增广Hookstep多步配对验证</title><style>body{margin:0;background:#edf2f7;color:#193047;font:17px/1.8 "Microsoft YaHei",sans-serif}main{max-width:1150px;margin:auto;padding:30px 22px}section{padding:28px;background:white;border-radius:10px;margin:24px 0}h1{font-size:32px}h2{font-size:25px;color:#155785}img{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;border:1px solid #ccd8e4}.scroll{overflow:auto}</style><main><h1>原生10 MHz：增广Hookstep多步配对验证</h1>'+''.join('<section id="s'+str(i+1)+'">'+s+'</section>' for i,s in enumerate(sections))
for name in ['A','B','protocol','endpoint_checks','linear_budget_check']:
    encoded=base64.b64encode((ROOT/(name+'.json')).read_bytes()).decode();body+=f'<p><a download="{name}.json" href="data:application/json;base64,{encoded}">下载{name}完整记录</a></p>'
(ROOT/'原生10MHz_增广Hookstep多步配对验证.html').write_text(body+'</main></html>',encoding='utf8')
