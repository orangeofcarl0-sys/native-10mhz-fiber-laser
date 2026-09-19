"""Chinese standalone evidence report: inner closure before outer energy tuning."""
import base64,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT,PROJECT
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':12})
read=lambda name:json.loads((ROOT/name).read_text(encoding='utf8'))
bench=read('support_benchmark.json');rows=bench['rows'];runs=read('deep_summary.json');protocol=read('deep_protocol.json');validation=read('factor_validation.json');checks=read('endpoint_checks.json')
def fig(name,draw):
    f,ax=plt.subplots(figsize=(11,5));draw(ax);ax.grid(alpha=.2);p=ROOT/(name+'.png');f.savefig(p,dpi=160,bbox_inches='tight');plt.close(f)
    return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
def table(headers,body):return '<div class="scroll"><table><tr>'+''.join('<th>'+v+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in body)+'</table></div>'
status_label={'iteration_budget_reached':'完成30步预算','residual_converged':'原方程残差达标','linear_accuracy_limited':'线性精度受限','globalization_stalled':'试探步未接受','radius_floor_reached':'信赖半径达下限'}
sections=[]
sections.append('<h2>本轮结果概览</h2><p>完整时间支持的C预条件器在同态基准中显著优于A/B；大Newton方向下还发现前向差分一致性问题，并用中心差分控制验证。正式固定泵浦求解中，27.7437、27.5、27.25、27.0 mW各完成30步，独立线性复核均低于1%，末态残差仍为1.29×10⁻³–1.43×10⁻³。27.75与28.0 mW因线性精度限制停止。</p><p><strong>本轮没有取得10⁻⁷周期一根，也未取得稳定单脉冲认证。</strong>六个末态的窗口加倍残差向量变化仅0.026%–0.049%。结果把线性误差、局部条件变差和非线性慢进展分开，但尚不能据此判定该family无根；未执行能量外层搜索。</p>')
sections.append('<h2>1．研究问题与本轮比较对象</h2><p>问题是27–28 mW附近能否闭合非零周期一方程。内层固定泵浦，仅求光场、EDF反转、相位及时间平移；输出能量作为观测量，不进入merit。只有内层取得并核验根，外层才可以将输出能量用于标量求根。</p><p>基准使用上一轮fixed_final末态，原方程残差'+f"{bench['initial_residual']:.7g}"+'，并非更早的0.001793状态。所有线性对照保持该状态、完整传播器、差分尺度和240维Arnoldi序列不变，从同一序列取120/180/240维前缀。独立中心差分复核最终方向，避免只看Hessenberg预测。</p><p>物理参数仍是20.42 m、CNT → OC、OC 80%、净色散+0.2 ps²，完整传播窗2.048 ns，dt=0.125 ps。目标是找到第一个根，尚未把0.9 nJ附近状态认作0.1–0.5 nJ工程目标已实现。</p>')
def benchmark_plot(ax):
    for label in ['A','B','C']:
        rr=[r for r in rows if r['support']==label];ax.semilogy([r['budget'] for r in rr],[100*r['central_relative'] for r in rr],marker='o',label=label)
    ax.axhline(1,color='gray',ls='--',label='1%');ax.set(xlabel='Arnoldi维数',ylabel='完整线性残差/%',title='同一状态下的线性维数与预条件支持对照');ax.legend()
text='<h2>2．线性维数与粗空间支持范围</h2><p>A：中心1.024 ns、±375 GHz；B：中心1.024 ns、±750 GHz；C：完整2.048 ns、±375 GHz。B和C均为6165维，A为3093维。仅改变预条件器中的投影和逆，原物理残差不截断。C的Fourier频率间隔随时间支持改变，因此不是简单嵌套的时间基底扩充。</p>'
text+=fig('同态线性残差',benchmark_plot)
text+=table(['支持','维数预算','前向复核/%','中心复核/%','Newton步范数','构建/s'],[[r['support'],r['budget'],f"{100*r['true_relative']:.5f}",f"{100*r['central_relative']:.5f}",f"{r['step_norm']:.5g}",f"{r['build_seconds']:.1f}"] for r in rows])
text+='<p>维数增加若继续降低残差，说明120维预算限制确实存在；若扩大支持降低投影外残差但线性精度变化有限，则不能把残差占比直接当成瓶颈贡献。不同状态的4.88%、7.71%等旧值不与本次末态重建值混用。C方案的Hessenberg预测可到约10⁻¹⁴，但独立映射复核仍约0.04%–0.06%，不能宣称机器精度闭合。</p>'
sections.append(text)
components=[next(r for r in rows if r['support']==s) for s in ['A','B','C']]
def partition(ax):
    xx=np.arange(3);a=np.array([r['temporal_relative']**2*100 for r in components]);b=np.array([r['spectral_relative']**2*100 for r in components]);ax.bar(xx,a,label='时间支持外');ax.bar(xx,b,bottom=a,label='时间支持内、频带外');ax.set_xticks(xx,['A','B','C']);ax.set(ylabel='占完整残差平方的比例/%',title='预条件器未覆盖残差的正交分解');ax.legend()
text='<h2>3．时间支持外与频带外残差的分解</h2><p>先把残差投影到指定时间区间，再对区间内部分作Fourier投影：R时间外=(I−T)R，R频带外=T(I−F)TR，剩余为粗空间内分量。三者正交，所以平方范数相加；8%和13%的范数比例不能直接相加。所有标量方程保留在粗空间。</p>'+fig('残差支持分解',partition)
text+=table(['支持','时间外范数/%','频带外范数/%','平方分量总和'],[[r['support'],f"{100*r['temporal_relative']:.4f}",f"{100*r['spectral_relative']:.4f}",f"{r['squared_partition']:.12f}"] for r in components])
text+='<p>预条件器时间支持和物理传播窗口是两个不同概念：C使用更宽的扰动/检测空间，并未改变传播模型；后续4.096 ns复核才是改变物理计算窗口。前一轮最弱20模泄漏不大，也不代表剩余子空间处理完全可靠。</p>'
sections.append(text)
text='<h2>4．重复构建粗矩阵时的分解方法</h2><p>同态基准使用原SVD逆。深求解采用同一粗矩阵的LU分解以降低重复SVD成本，不修改物理方程。估计条件数过高或奇异时回退SVD，并保存粗方程回代误差。初始每3步重建的短试跑保存在归档static_rebuild_pilot目录，正式数据统一从原末态按精度触发刷新协议重启，未混入短试跑步骤。先在同一真实状态核对LU与SVD的240维线性结果，合格才开始深求解。</p>'
text+=table(['支持','LU完整线性残差','SVD完整线性残差','粗方程回代误差','分解构建/s'],[[validation['support'],f"{validation['LU_linear_relative']:.7g}",f"{validation['SVD_linear_relative']:.7g}",f"{validation['factor_probe_residual']:.3g}",f"{validation['factor_seconds']:.1f}"]])
text+='<p>LU条件估计使用1范数，旧SVD条件数使用2范数，不能直接比较两者数值。初始分解和验证调用单独记录，随后首次内层求解复用该分解；不是把构建成本当成零。</p>'
sections.append(text)
jvp=read('jvp_consistency.json')
text='<h2>5．大Newton方向下的导数一致性检查</h2><p>前向差分试跑在锚点接受两步后，完整Newton方向范数达到2418.6。Arnoldi预测线性残差为0.78%，但刷新粗矩阵后直接作用于该方向得到11.06%，因此没有接受该步。其余泵浦点未接受场更新，能量相同不是物理分支的证据。下表保持同一停止态和预条件器，仅改变方向导数算法；最后再用不同扰动尺度的中心差分复核。</p>'
text+=table(['Jv算法','扰动范数','维数','预测/%','同算法复核/%','中心10⁻⁴/%','中心10⁻⁵/%','中心10⁻⁶/%','方向范数'],[[r['method'],r['epsilon'],r['krylov'],f"{100*r['predicted']:.4g}",f"{100*r['own_relative']:.4g}",*[f"{100*v['relative']:.4g}" for v in r['checks']],f"{r['step_norm']:.5g}"] for r in jvp['rows']])
def derivative_plot(ax):
    for r in jvp['rows']:ax.loglog([v['epsilon'] for v in r['checks']],[100*v['relative'] for v in r['checks']],marker='o',label=f"{r['method']} h={r['epsilon']:g}")
    ax.axhline(1,color='gray',ls='--');ax.set(xlabel='独立中心差分的状态扰动范数',ylabel='完整线性残差/%',title='同一停止态的方向导数一致性');ax.legend()
text+=fig('方向导数一致性',derivative_plot)
text+='<p>状态扰动范数h与Jv中实际倍乘系数不同：实际系数=h/‖v‖。组合方向很大时，差分误差会被放大。只有跨尺度复核一致，才能把小Hessenberg残差当成可信的线性解；本状态中心差分h=10⁻⁵的跨尺度复核为0.78%–0.96%，而前向解的中心复核约14%。这支持差分误差被大方向放大的解释。后续正式求解从原fixed_final重新开始，用中心h=10⁻⁵、独立h=10⁻⁶检查，不混入前向试跑步骤。此项是数值诊断，不是光纤物理不稳定性的证明。</p>'
sections.append(text)
full=[read(r['name']+'.json') for r in runs]
def histories(ax):
    for r in full:ax.semilogy([v['step'] for v in r['history']],[v['residual'] for v in r['history']],marker='o',markersize=2.5,label=f"{r['pump_W']*1000:.4f} mW")
    ax.set(xlabel='该泵浦点的Newton–Hookstep步骤',ylabel='原方程残差',title='逐级热启动的固定泵浦闭合历史');ax.legend()
text='<h2>6．27–28 mW逐级热启动与内层收敛记录</h2><p>先从27.7437 mW末态进一步求解，再向下27.5→27.25→27.0 mW；向上从锚点结果另起27.75→28.0 mW。父状态关系在表中列明。未收敛状态也可作为数值初猜，但这些连接不是已认证的稳态分支。</p>'
text+=f"<p>统一协议：支持{protocol['support']}，最多{protocol['linear_limit']}维，Arnoldi目标{100*protocol['linear_tolerance']:.2f}%，中心差分扰动范数{protocol['jv_epsilon']:g}，独立复核尺度{protocol['jv_check_epsilon']:g}，每点最多{protocol['max_steps']}步，初始半径{protocol['initial_radius']}，定期重建上限{protocol['rebuild_every']}步；实际线性误差超过1%时，在同一状态刷新粗矩阵并重算，刷新后仍不达标则以线性精度受限停止，不接受该步。不启用旧的2%进展提前停止；残差低于10⁻⁷才按根停止。实际线性误差是否小于1%另行统计，不能用请求的容差替代达到的精度。</p>"
text+=fig('深求解残差历史',histories)
text+=table(['泵浦/mW','父状态','停止状态','步数','原方程残差','输出/nJ','线性<1%比例'],[[f"{r['pump_W']*1000:.4f}",(next((f"{v['pump_W']*1000:.4f} mW" for v in runs if v['name']==r['parent']),'上轮末态')),status_label.get(r['status'],r['status']),r['history'][-1]['step'],f"{r['physical_residual']:.6g}",f"{r['output_energy_nJ']:.6f}",f"{100*r['linear_fraction_below_one_percent']:.1f}%"] for r in full])
text+=table(['泵浦/mW','末5步残差下降/%','末步信赖半径','最大已复核线性残差/%','精度触发重建次数'],[[f"{r['pump_W']*1000:.4f}",f"{100*(1-r['history'][-1]['residual']/r['history'][max(0,len(r['history'])-6)]['residual']):.4f}",f"{r['history'][-1]['radius']:.5g}",f"{100*max(v.get('true_newton_linear_residual',0) for v in r['history']):.4f}",sum(v.get('refreshed_for_accuracy',False) for v in r['history'])] for r in full])
def steps_plot(ax):
    hh=[v for v in full[0]['history'] if 'accepted_trial' in v]
    ax.semilogy([v['step'] for v in hh],[v['newton_blocks']['total'] for v in hh],label='完整Newton方向')
    ax.semilogy([v['step'] for v in hh],[v['trials'][v['accepted_trial']]['step_blocks']['total'] for v in hh],label='实际接受的Hookstep')
    ax.set(xlabel='锚点Newton–Hookstep步骤',ylabel='缩放状态步长范数',title='锚点完整Newton方向与实际接受步长');ax.legend()
text+=fig('锚点步长对照',steps_plot)
text+='<p>线性方程解得准确，并不意味着完整Newton步可直接接受；信赖域仍限制实际步长。末五步下降率在记录不足五步时使用已有步骤（零步记为0）；它描述有限预算末端的进展，不是驻点证明，也不是不存在根的证明。此处的迭代步数不是物理腔内往返次数。</p>'
text+=table(['泵浦/mW','最后使用的分解','最后粗矩阵1范数条件估计','最后Newton方向范数'],[[f"{r['pump_W']*1000:.4f}",next(v['factorization'] for v in reversed(r['history']) if 'factorization' in v),f"{next(v['coarse_condition_1_estimate'] for v in reversed(r['history']) if 'coarse_condition_1_estimate' in v):.4g}",f"{next(v['newton_blocks']['total'] for v in reversed(r['history']) if 'newton_blocks' in v):.5g}"] for r in full])
text+='<p>C是同一初始态三个受测方案中线性表现最好的方案，不是全程最优性证明。27.75 mW第6步的粗矩阵1范数条件估计升至约1.6×10⁹，触发SVD回退，完整Newton方向范数约1.86万，独立线性复核为2.88%，所以停止。不能把该点列入“线性充分准确却仍不收敛”的证据。</p>'
text+='<p>不把这些最终残差连成“真实残差极小值—泵浦曲线”，也不把未收敛输出的差商当成稳态dE/dP。有限步结果仅回答本协议和该热启动路径实际获得了什么。</p>'
sections.append(text)
def linear(ax):
    for r in full:
        hh=[v for v in r['history'] if 'true_newton_linear_residual' in v];ax.semilogy([v['step'] for v in hh],[100*v['true_newton_linear_residual'] for v in hh],marker='o',markersize=3,label=f"{r['pump_W']*1000:.4f} mW")
    ax.axhline(1,color='gray',ls='--');ax.set(xlabel='Newton–Hookstep步骤',ylabel='实际完整线性残差/%',title='内层线性精度的逐步验收');ax.legend()
text='<h2>7．线性精度与停止态波形</h2>'+fig('深求解线性精度',linear)
for r in full:
    z=np.load(ROOT/(r['name']+'.npz'))
    def pulse(ax):
        p=np.sum(abs(z['output'])**2,axis=0);t=(np.arange(len(p))-len(p)/2)*float(z['dt']);ax.plot(t,p);ax.set(xlabel='时间/ps',ylabel='输出功率/W',title=f"{r['pump_W']*1000:.4f} mW停止态输出")
    text+=fig(r['name']+'_pulse',pulse)
text+='<p>每个波形独立大图；局部峰值数量不自动等于分离脉冲数量。低残差、非零能量、局域形态、数值收敛及动态稳定性是不同验收项。</p>'
sections.append(text)
text='<h2>8．窗口复核、能量外层条件与结论边界</h2>'
text+=table(['状态','原窗口残差','4.096 ns残差','完整残差向量相对变化'],[[r['name'],f"{r['physical_residual']:.6g}",f"{r['wide_residual']:.6g}",f"{100*r['full_vector_relative_change']:.4f}%"] for r in checks])
text+='<p>能量外层只能调用已取得并核验非零根的内层结果；未收敛时不执行Brent或割线求解，不将0.9 nJ硬约束重新加入merit。上一轮能量行初始占98.91%平方残差，这说明缩放改变了全局化重点，不证明0.9 nJ与周期一根不相容。</p>'
text+=f"<p>本轮达到原方程10⁻⁷门槛的点数：{sum(r['numerical_root'] for r in runs)}。即使达到门槛，仍须检查时间步、传播步长、非零局域性；稳定性另需物理动力学/Floquet证据。线性精度未达到要求或有限预算结束时，不能将未找到根升级为无根结论。</p>"
sections.append(text)
body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：固定泵浦线性精度与窄区间求根</title><style>body{margin:0;background:#edf2f7;color:#193047;font:17px/1.8 "Microsoft YaHei",sans-serif}main{max-width:1150px;margin:auto;padding:30px 22px}section{padding:28px;background:white;border-radius:10px;margin:24px 0}h1{font-size:32px}h2{font-size:25px;color:#155785}img{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;border:1px solid #ccd8e4}.scroll{overflow:auto}</style><main><h1>原生10 MHz：固定泵浦线性精度与窄区间求根</h1>'+''.join('<section>'+s+'</section>' for s in sections)
for name in ['support_benchmark','factor_validation','jvp_consistency','deep_protocol','deep_summary','endpoint_checks']+[r['name'] for r in runs]:
    encoded=base64.b64encode((ROOT/(name+'.json')).read_bytes()).decode();body+=f'<p><a download="{name}.json" href="data:application/json;base64,{encoded}">下载{name}记录</a></p>'
(ROOT/'原生10MHz_固定泵浦线性精度与窄区间求根.html').write_text(body+'</main></html>',encoding='utf8')
