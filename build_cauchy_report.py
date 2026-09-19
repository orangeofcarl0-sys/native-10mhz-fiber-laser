"""Offline report for fixed-state span audit and Cauchy enrichment."""
import json,base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':12})
runs=[json.loads((ROOT/(name+'.json')).read_text(encoding='utf8')) for name in ['down_270','down_275']]

def table(head,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+h+'</th>' for h in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'
def fig(name,draw):
    f,ax=plt.subplots(figsize=(11,5));draw(ax);ax.grid(alpha=.2);p=ROOT/(name+'.png');f.savefig(p,dpi=160,bbox_inches='tight');plt.close(f)
    return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
def pct(v):return '—' if v is None else f'{100*v:.5g}%'
sections=[]
sections.append('<h2>1．研究对象与四项同态对照</h2><p>只使用上一轮27.0/27.5 mW保存状态，研究已知Cauchy方向为何优于原Hookstep，以及把该方向加入后能否满足模型下降要求。所有试探从同一初值出发，不累计更新，不运行30步续算。模型仍是20.42 m、CNT → OC、OC 80%、净色散+0.2 ps²，窗口2.048 ns、dt=0.125 ps；泵浦固定，能量不进入merit。</p><p>A：原始Krylov空间与whitening保留空间的方向覆盖；B：将Cauchy步投影回原空间并核对模型；C：Hookstep与Cauchy按模型择优；D：在原基底加入Cauchy方向后求解。三个半径为0.00156、0.003125、0.00625，均指缩放状态范数，D=I。</p><p>预条件器保持前向h=10⁻⁶的C配置，Krylov Jv用中心h=10⁻⁵，独立复核h=10⁻⁶。梯度方向来自上一轮完整输出审计，输入SHA完全相同；这里不是跨状态复用陈旧梯度。</p>')
text='<h2>2．原始Krylov空间与whitening后的方向覆盖</h2><p>Z是右预条件Arnoldi的实际状态方向，d_C是已验证的单位受限负梯度方向。用SVD正交投影计算ε=‖(I−QQᵀ)Dd_C‖/‖Dd_C‖，χ=‖QᵀDd_C‖/‖Dd_C‖。ε²+χ²应为1。原始空间采用相对阈值10⁻¹⁴，whitening保留空间采用生产阈值10⁻¹²；完整奇异值保存在NPZ中。</p>'
text+=table(['泵浦/mW','空间','维数','ε','χ','ε²+χ²','Z最小/最大奇异值'],[[r['pump_W']*1000,v['span'],v['rank'],f"{v['epsilon']:.7g}",f"{v['coverage']:.7g}",f"{v['partition']:.12f}",f"{r['smallest_relative_state_singular']:.5g}"] for r in runs for v in r['span']])
def spans(ax):
    xx=np.arange(2)
    for j,kind in enumerate(['raw','kept']):ax.bar(xx+(j-.5)*.25,[100*r['span'][j]['coverage'] for r in runs],width=.25,label=kind)
    ax.set_xticks(xx,[f"{r['pump_W']*1000:g} mW" for r in runs]);ax.set(ylabel='Cauchy方向投影长度占比 χ/%',title='原始与保留状态子空间的方向覆盖');ax.legend()
text+=fig('子空间方向覆盖',spans)
text+='<p>只有在原始空间已能表示d_C、但保留空间不能表示时，才支持whitening截断解释。如果两者都遗漏很多，优先定位为原始搜索空间的限制。这里是两个特定状态的方向覆盖，不是完整Jacobian非正规性或全局求解性质的证明。</p>'
sections.append(text)
text='<h2>3．投影Cauchy步的可行性与模型一致性</h2><p>先求系数y_C使Zy_C尽量接近s_C，再比较‖βe₁−Hy_C‖与独立完整映射差分得到的‖R+J(Zy_C)‖。若重构误差很大，后者必须使用重构后的Zy_C，不能拿原s_C的模型值直接比较并据此宣称小型求解器错误。</p>'
text+=table(['泵浦/mW','Cauchy步重构相对误差','重构步范数','reduced模型残差','独立完整模型残差','范数差/初始R'],[[r['pump_W']*1000,f"{r['trials'][0]['representation']['step_relative_error']:.6g}",f"{r['trials'][0]['representation']['represented_step_norm']:.6g}",f"{r['trials'][0]['representation']['reduced_norm']:.9g}",f"{r['trials'][0]['representation']['independent_norm']:.9g}",pct(r['trials'][0]['representation']['relative_model_norm_difference'])] for r in runs])
text+=table(['泵浦/mW','Δ','投影Cauchy预测下降','原Hookstep预测下降','原Hookstep不劣于可行投影候选'],[[r['pump_W']*1000,v['radius'],f"{v['representation']['represented_prediction']:.7g}",f"{v['hookstep']['model_prediction']:.7g}",'是' if v['hookstep']['model_prediction']>=v['representation']['represented_prediction']*(1-1e-6) else '否'] for r in runs for v in r['trials']])
text+='<p>三个半径均大于未截断Cauchy步，投影审计相同，所以表中只列一次。另有已知答案测试：当Cauchy确实位于原空间时，原Hookstep必须不劣于它；不满足将判为测试失败，而不是归因于空间缺失。</p>'
sections.append(text)
text='<h2>4．Cauchy择优与梯度增广的模型下降</h2><p>Cauchy步长取α=min(−RᵀJd/‖Jd‖², Δ/‖Dd‖)，必须先验证RᵀJd&lt;0。择优方案比较原Hookstep和Cauchy的完整模型预测。增广方案保留原Z、V、H，将d和Jd加入；用V对Jd做两次正交化，构成新的小型输出矩阵，再调用相同的metric-whitening Hookstep。</p><p>硬验收：原始增广步Δm_aug ≥ Δm_C(1−10⁻⁶)。运行时若模型或半径验收失败，回退Cauchy并记录失败标志；本报告中的增广结果始终指回退之前的原始步，不能靠回退伪装通过。</p>'
for r in runs:
    def model(ax):
        xx=np.arange(3)
        for j,(key,label) in enumerate([('hookstep','原Hookstep'),('cauchy','Cauchy'),('augmented','梯度增广')]):ax.bar(xx+(j-1)*.24,[v[key]['model_prediction']*1e9 for v in r['trials']],width=.24,label=label)
        ax.set_xticks(xx,[str(v['radius']) for v in r['trials']]);ax.set(xlabel='信赖半径Δ',ylabel='预测merit下降 (单位: 1e-9)',title=f"{r['pump_W']*1000:g} mW三类候选的模型下降");ax.legend()
    text+=fig(r['name']+'_model',model)
    text+=table(['Δ','择优选择','增广/Cauchy预测下降','原始模型验收','是否回退','独立增广预测/模型预测'],[[v['radius'],v['safeguard_choice'],f"{v['augmented']['model_prediction']/v['cauchy']['model_prediction']:.7g}",'通过' if v['guard']['raw_model_pass'] else '失败','是' if v['guard']['fallback'] else '否',f"{v['augmented']['independent_prediction']/v['augmented']['model_prediction']:.9g}"] for v in r['trials']])
sections.append(text)
text='<h2>5．完整腔映射的实际下降与接受比</h2><p>对每个原始候选独立计算R(x+s)，再给出ρ=实际merit下降/预测下降。硬验收还要求ρ_aug&gt;0.1；实际残差低于Cauchy是单独的更强结果，不能由模型保证自动推出。</p>'
for r in runs:
    def actual(ax):
        xx=np.arange(3)
        for j,(key,label) in enumerate([('hookstep','原Hookstep'),('cauchy','Cauchy'),('augmented','梯度增广')]):ax.bar(xx+(j-1)*.24,[100*v[key]['residual_decrease'] for v in r['trials']],width=.24,label=label)
        ax.set_xticks(xx,[str(v['radius']) for v in r['trials']]);ax.set(xlabel='信赖半径Δ',ylabel='实际完整残差下降/%',title=f"{r['pump_W']*1000:g} mW单步完整映射结果");ax.legend()
    text+=fig(r['name']+'_actual',actual)
    text+=table(['Δ','原Hookstep下降','Cauchy下降','择优下降','增广下降','增广ρ','实际优于Cauchy'],[[v['radius'],pct(v['hookstep']['residual_decrease']),pct(v['cauchy']['residual_decrease']),pct(v['safeguard']['residual_decrease']),pct(v['augmented']['residual_decrease']),f"{v['augmented']['rho']:.6g}",'是' if v['acceptance']['beats_cauchy_actual'] else '否'] for v in r['trials']])
sections.append(text)
allrows=[v for r in runs for v in r['trials']]
text='<h2>6．验收汇总与结论边界</h2>'+table(['验收项','通过/总数'],[['原始增广模型不劣于Cauchy',f"{sum(v['acceptance']['model_cauchy'] for v in allrows)}/6"],['增广实际ρ>0.1',f"{sum(v['acceptance']['actual_rho'] for v in allrows)}/6"],['增广实际残差优于Cauchy',f"{sum(v['acceptance']['beats_cauchy_actual'] for v in allrows)}/6"],['触发回退次数',sum(v['guard']['fallback'] for v in allrows)]])
text+='<p id="interpretation">两个状态的Cauchy方向投影长度占比仅为0.538%和0.411%；原始/保留空间完全一致，37/31个方向全部保留，支持原Krylov搜索空间遗漏方向的诊断，不支持本次whitening截断解释。投影候选的reduced模型与独立完整模型相符，原Hookstep优于该可行候选，本次没有发现小型优化器违反可行候选下界。加入方向后六个原始增广步全部通过，ρ为0.950–0.999，未触发回退。27.0/27.5 mW实际残差单步下降分别为0.639–0.659%和1.180–1.206%，超过单独Cauchy的0.495%和0.925%。运行时保护已在增广步函数实现并测试；本轮未改变长期求解器的默认流程。</p><p>此次仅证明受测状态下的算法方向/模型性质。受限Cauchy方向不是完整JᵀR；该保障不等于全空间全局收敛定理。没有推进30步，没有新增周期一根、稳定单脉冲或period-p证据。</p><p>参考：<a href="https://optimization-online.org/wp-content/uploads/2007/11/1840.pdf">Erway等的trust-region子问题论文</a>讨论低维子空间中的信赖域步；<a href="https://ralna.github.io/galahad_docs/html/C/gltr.html">GALAHAD GLTR文档</a>说明对称二次信赖域求解。这里仍使用GMRES根求解子空间，并仅增广一个已验证方向，没有把当前实现称为完整Gauss–Newton或GLTR。</p>'
sections.append(text)
sections.append('<h2>7．实现与复现记录</h2><p>steady_cauchy.py提供子空间审计、Cauchy步和带显式失败标志的增广步；steady_hookstep.arnoldi只新增可选return_basis，默认四个返回值保持兼容。小规模测试包括缺失方向、whitening截断、可行Cauchy候选、增广模型保证和强制失败回退。</p><p>配置CuPy及独立LASER_OUTPUT_DIR，运行run_cauchy_audit.py，再运行build_cauchy_report.py。JSON记录输入/梯度源SHA和执行源码SHA；公开NPZ保留方向、投影系数、奇异值和全部试探步。本地*_basis.npz保存完整Z、V、H，SHA写入JSON；大型基底不随公开精简结果上传，可用同一输入和脚本重建。</p>')
body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：Cauchy增广Hookstep与子空间审计</title><style>body{margin:0;background:#edf2f7;color:#193047;font:17px/1.8 "Microsoft YaHei",sans-serif}main{max-width:1150px;margin:auto;padding:30px 22px}section{padding:28px;background:white;border-radius:10px;margin:24px 0}h1{font-size:32px}h2{font-size:25px;color:#155785}img{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;border:1px solid #ccd8e4}.scroll{overflow:auto}</style><main><h1>原生10 MHz：Cauchy增广Hookstep与子空间审计</h1>'+''.join('<section id="s'+str(i+1)+'">'+s+'</section>' for i,s in enumerate(sections))
for r in runs:
    name=r['name'];encoded=base64.b64encode((ROOT/(name+'.json')).read_bytes()).decode();body+=f'<p><a download="{name}.json" href="data:application/json;base64,{encoded}">下载{name}完整诊断记录</a></p>'
(ROOT/'原生10MHz_Cauchy增广Hookstep与子空间审计.html').write_text(body+'</main></html>',encoding='utf8')
