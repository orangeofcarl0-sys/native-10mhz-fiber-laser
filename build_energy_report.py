"""Offline report of leakage controls and physical energy closure."""
import base64,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':12})
read=lambda name:json.loads((ROOT/name).read_text(encoding='utf8'))
def figure(name,draw):
    fig,ax=plt.subplots(figsize=(11,5));draw(ax);ax.grid(alpha=.2)
    p=ROOT/(name+'.png');fig.savefig(p,dpi=160,bbox_inches='tight');plt.close(fig)
    return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
def table(headers,rows):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+v+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
status_label={'iteration_budget_reached':'完成步数预算','progress_stagnated':'触发进展不足规则','residual_converged':'残差达到求根门槛','globalization_stalled':'试探步未接受','radius_floor_reached':'半径达到下限'}
sections=[]
sections.append('<h2>1．研究问题与对照条件</h2><p>工程目标仍为0.1–0.5 nJ，0.9 nJ只是邻近当前初猜的求根试验锚点。本轮检验投影外响应是否造成27.744 mW末态的线性求解困难，并用固定泵浦和固定输出能量重新寻找非零周期一解。物理模型保持20.42 m、CNT → OC、OC 80%、净色散+0.2 ps²；时间窗2.048 ns，dt=0.125 ps。</p><p>此前50 mW固定参数系统的最弱投影方向有约244倍完整响应，但该数值不能直接代替新状态、新方程的泄漏诊断。原泵浦轨迹采用人为超平面，未取得精确根，因此不是已认证的稳态分支。</p>')
for name,title in [('leakage_control.json','保留原超平面的增广系统'),('leakage_fixed_control.json','删除超平面的固定泵浦系统')]:
    data=read(name);rows=data['rows'];modes=rows[0]['leakage_modes']
    def draw(ax):
        ax.semilogy(range(1,21),[v['leakage'] for v in modes[::-1]],marker='o')
        for lim in [5,10,20]:ax.axhline(lim,ls='--',alpha=.5,label='门槛 '+str(lim))
        ax.set(xlabel='最弱方向序号（1最弱）',ylabel='投影外响应 / 投影内响应',title=title+'：最弱20模泄漏比');ax.legend()
    text='<h2>'+str(len(sections)+1)+'．'+title+'的同态线性对照</h2>'
    text+='<p>对同一状态重建Jc=PJE，E是中心窗口Fourier提升，P=Eᵀ。用独立中心差分计算d=JEv及L=‖d−EPd‖/‖Pd‖。只审核最弱20个方向；这不排除其他未审核方向的问题。</p>'
    text+='<p>所有对照使用120维Arnoldi预算、3%停止目标、同一完整残差与差分尺度。baseline保留原逆；truncate将超门槛模式逆系数置零；cap将这些模式的逆系数限制为1/max(σ,‖JEv‖)。截断会使预条件器奇异，可能丢失修正方向，因此必须以完整线性残差验证，不能预先认定改进。</p>'
    text+=figure(name.replace('.json',''),draw)
    text+=table(['方式','L门槛','改动模式数','Krylov维数','完整线性相对残差','Newton步范数'],[[{'baseline':'原逆','truncate':'泄漏截断','cap':'响应限幅'}[r['mode']],r['threshold'],r['modified_modes'],r['krylov'],f"{100*r['true_residual']:.5f}%",f"{r['step_norm']:.5g}"] for r in rows])
    text+=f"<p>当前最弱20模最大L={max(v['leakage'] for v in modes):.5g}。残差以完整JΔx+R独立重算；当前末态重建值不应与此前倒数一步、旧预条件器的3.85%混为同一次测量。不能把未触发的模式规则称为已改善预条件器。</p>"
    sections.append(text)
profiles=[read(n+'.json') for n in ['fixed_40','fixed_35','fixed_30','fixed_final']]
text='<h2>4．固定泵浦条件下的有限步求根</h2><p>删除参数未知量和超平面方程，固定泵浦40、35、30、27.744 mW，各最多8步，信赖半径0.1，五步进展规则与之前相同。40 mW使用保存的40.0001 mW状态；其余使用同一个27.744 mW状态。历史35/30 mW场未保存，本轮不将这些共同初猜冒充原轨迹快照。</p><p>所以结果是指定初猜、有限预算获得的残差，不是全局或已证明的局部最小值；40 mW与其余点还存在初猜差别。仅当受控趋势足够一致时才讨论泵浦趋势。</p>'
def profiles_plot(ax):
    for p in profiles:ax.semilogy([v['step'] for v in p['history']],[v['residual'] for v in p['history']],marker='o',label=f"{p['pump_W']*1000:.3f} mW")
    ax.set(xlabel='Hookstep步骤',ylabel='完整原方程残差',title='固定泵浦求根历史');ax.legend()
text+=figure('固定泵浦残差历史',profiles_plot)
def profile_endpoint(ax):
    common=sorted(profiles[1:],key=lambda p:p['pump_W'])
    ax.semilogy([p['pump_W']*1000 for p in common],[p['physical_residual'] for p in common],marker='o',label='共同27.744 mW初猜')
    ax.semilogy([profiles[0]['pump_W']*1000],[profiles[0]['physical_residual']],marker='s',ls='',label='独立40 mW初猜')
    ax.set(xlabel='固定泵浦/mW',ylabel='有限步获得的原方程残差',title='泵浦与有限步求根结果（不是稳态分支）');ax.legend()
text+=figure('固定泵浦结果对照',profile_endpoint)
text+=table(['泵浦/mW','停止','步骤','原方程残差','相对场残差','输出/nJ','峰数'],[[f"{p['pump_W']*1000:.4f}",status_label.get(p['status'],p['status']),p['history'][-1]['step'],f"{p['physical_residual']:.6g}",f"{p['relative_field_residual']:.6g}",f"{p['output_energy_nJ']:.6f}",p['peaks']] for p in profiles])
for run in profiles:
    saved=np.load(ROOT/(run['label']+'.npz'))
    def draw_profile(ax):
        power=np.sum(abs(saved['output'])**2,axis=0);t=(np.arange(len(power))-len(power)/2)*float(saved['dt'])
        ax.plot(t,power);ax.set(xlabel='时间/ps',ylabel='输出瞬时功率/W',title=f"固定{run['pump_W']*1000:.3f} mW停止态输出（非稳定认证）")
    text+=figure(run['label']+'_output',draw_profile)
text+='<p>峰数按功率峰prominence≥全局峰值10%计数；35 mW的第二局部峰位于主脉冲前沿，不能将峰值计数直接当作分离脉冲数。局部尖峰必须单列，不把能量或残差较低等同于稳定单脉冲。</p>'
text+='<p>默认选择规则：只有固定泵浦同态对照的完整线性残差比baseline改善至少5%，才切换预条件器；否则保留原实现。该规则只控制本轮试验，不是普遍最优标准。</p>'
sections.append(text)
e=read('energy_09.json')
energy_trace=e['trace']+[dict(step=e['history'][-1]['step'],physical_residual=e['physical_residual'],energy_equation=e['energy_relative_error'])]
text='<h2>5．0.9 nJ输出能量约束与自由泵浦</h2><p>未知量仍为光场、EDF反转、整体相位、时间平移及泵浦，但最后一个方程改为(Eout−0.9 nJ)/(0.9 nJ)=0。Eout来自同一完整腔传播的输出耦合端，按Σ|Aout|² dt/1000将W·ps转成nJ，不使用输入场能量替代。</p><p>正能量约束排除零场精确解，不保证周期一解存在，也不保证找到的解动态稳定。标量以目标能量归一化；它参与merit和信赖域，因而其权重也是求解设计的一部分。泵浦限制5–60 mW，从原27.744 mW末态开始，最多20步，未直接跳到0.5 nJ。</p>'
def energy_plot(ax):
    ax.semilogy([v['step'] for v in e['history']],[v['residual'] for v in e['history']],marker='o',label='含能量方程的增广残差')
    ax.semilogy([v['step'] for v in energy_trace],[v['physical_residual'] for v in energy_trace],marker='.',label='原方程残差')
    ax.set(xlabel='Hookstep步骤',ylabel='残差',title='0.9 nJ约束求根历史');ax.legend()
text+=figure('能量约束求根历史',energy_plot)
initial_physical=e['trace'][0]['physical_residual']
text+=f"<p><strong>本次观察：</strong>输出能量逼近目标，但原方程残差从{initial_physical:.6g}变为{e['physical_residual']:.6g}，变化{100*(e['physical_residual']/initial_physical-1):+.2f}%。因此增广merit下降不能被表述为周期闭合改善；本次能量约束尚未产生精确锚点。停止是完成20步预算，末段仍下降，不证明该约束下无根。</p>"
def energy_constraint(ax):
    ax.plot([v['step'] for v in energy_trace],[.9*(1+v['energy_equation']) for v in energy_trace],marker='o')
    ax.axhline(.9,color='gray',ls='--',label='目标0.9 nJ')
    ax.set(xlabel='Hookstep步骤',ylabel='输出能量/nJ',title='能量方程的满足程度');ax.legend()
text+=figure('能量方程满足程度',energy_constraint)
text+=table(['停止','泵浦/mW','原方程残差','增广残差','能量相对误差','输出/nJ'],[[status_label.get(e['status'],e['status']),f"{e['pump_W']*1000:.5f}",f"{e['physical_residual']:.6g}",f"{e['augmented_residual']:.6g}",f"{e['energy_relative_error']:.6g}",f"{e['output_energy_nJ']:.6f}"]])
state=np.load(ROOT/'energy_09.npz')
def pulse(ax):
    p=np.sum(abs(state['output'])**2,axis=0);t=(np.arange(len(p))-len(p)/2)*float(state['dt'])
    ax.plot(t,p);ax.set(xlabel='时间/ps',ylabel='输出瞬时功率/W',title='能量约束试验停止态输出')
text+=figure('能量约束停止态输出',pulse)
text+=f"<p>局部峰数{e['peaks']}，相对场闭合残差{e['relative_field_residual']:.6g}，时间边缘能量比例{e['time_edge']:.4g}，频谱边缘比例{e['spectral_edge']:.4g}。峰数只是形态诊断，不能替代稳态/稳定性验收。</p>"
if (ROOT/'energy_endpoint_check.json').exists():
    check=read('energy_endpoint_check.json')
    text+=f"<p>GPU批量/串行相对差{check['batch_serial_relative']:.3g}；能量方向导数步长减半相对差{check['energy_derivative_halving_relative']:.3g}。窗口加宽至4.096 ns后原方程残差{check['wide_physical_residual']:.6g}，完整增广残差向量相对变化{check['full_vector_relative_change']:.4g}。窗口检查不能替代时间步与传播步长验收。</p>"
sections.append(text)
def linear_history(ax):
    for run in profiles+[e]:
        values=[v for v in run['history'] if 'true_newton_linear_residual' in v]
        ax.semilogy([v['step'] for v in values],[100*v['true_newton_linear_residual'] for v in values],marker='.',label=('0.9 nJ约束' if run['label']=='energy_09' else f"{run['pump_W']*1000:.3f} mW"))
    ax.axhline(3,color='gray',ls='--');ax.set(xlabel='Hookstep步骤',ylabel='完整线性相对残差/%',title='各试验的线性求解精度');ax.legend()
def radius_history(ax):
    for run in profiles+[e]:ax.semilogy([v['step'] for v in run['history']],[v['radius'] for v in run['history']],marker='.',label=('0.9 nJ约束' if run['label']=='energy_09' else f"{run['pump_W']*1000:.3f} mW"))
    ax.set(xlabel='Hookstep步骤',ylabel='缩放状态中的信赖半径',title='各试验的步长限制');ax.legend()
sections.append('<h2>6．线性求解与信赖域诊断</h2><p>线性残差是‖JΔx+R‖/‖R‖，不是非线性原方程残差。虚线3%为Arnoldi目标；达到120维仍高于该值表示预算内未满足目标。信赖半径作用于已缩放的场、反转、相位、时移和参数整体步长，不是物理传播时间。</p>'+figure('各试验线性残差',linear_history)+figure('各试验信赖半径',radius_history))
sections.append('<h2>7．证据边界与复现</h2><p>原方程残差门槛为10⁻⁷。只有取得非零根并完成网格、窗口和原映射核验，才推进0.8→0.7→0.6→0.5 nJ或正式伪弧长延拓。本报告中的预算停止不证明无根，输出接近目标也不等于方程闭合。</p><p>50项CPU回归通过；新增测试覆盖原版/新版基线预条件器一致、同批/串行输出能量一致、W·ps到nJ单位、零场能量方程为−1及投影伪软模的截断/限幅行为。独立JSON和末态NPZ保留全部迭代，报告图为单独大图并内嵌，历史报告保持不变。</p>')
body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：投影泄漏与固定能量求根</title><style>body{background:#edf2f7;color:#193047;font:17px/1.8 "Microsoft YaHei",sans-serif;margin:0}main{max-width:1150px;margin:auto;padding:30px 22px}section{background:white;padding:28px;margin:24px 0;border-radius:10px}h1{font-size:32px}h2{font-size:25px;color:#155785}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd8e4;padding:9px}.scroll{overflow:auto}</style><main><h1>原生10 MHz：投影泄漏与固定能量求根</h1>'+''.join('<section>'+s+'</section>' for s in sections)
for name in ['leakage_control','leakage_fixed_control','fixed_40','fixed_35','fixed_30','fixed_final','energy_09']:
    data=base64.b64encode((ROOT/(name+'.json')).read_bytes()).decode()
    body+=f'<p><a download="{name}.json" href="data:application/json;base64,{data}">下载{name}完整记录</a></p>'
(ROOT/'原生10MHz_投影泄漏与固定能量求根.html').write_text(body+'</main></html>',encoding='utf8')
