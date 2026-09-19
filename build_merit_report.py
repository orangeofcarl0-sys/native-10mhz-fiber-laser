"""Standalone Chinese report for three fixed-state numerical controls."""
import base64,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':12})
read=lambda name:json.loads((ROOT/name).read_text(encoding='utf8'))
low=[read(n+'.json') for n in ['down_270','down_275']];up=read('up_2775.json')

def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
def figure(name,draw):
    f,ax=plt.subplots(figsize=(11,5));draw(ax);ax.grid(alpha=.2);p=ROOT/(name+'.png');f.savefig(p,dpi=160,bbox_inches='tight');plt.close(f)
    return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
def pct(v):return '—' if v is None else f'{100*v:.5g}%'
sections=[]
sections.append('<h2>本轮结果概览</h2><p><strong>当前数据不支持把27.0/27.5 mW末态判为非零残差极小值。</strong>两点都存在导数复核一致的负梯度步，完整残差分别下降0.495%和0.925%，优于本次同态Hookstep扫描的0.166%和0.284%。两点将半径增至0.025或0.05均变坏，说明简单放大半径不是答案。</p><p>27.75 mW的中心粗矩阵对照没有实质改善：κ₁约1.57×10⁹→1.58×10⁹，同尺度完整线性复核2.88%→2.85%。本轮仅做同态诊断，没有新增周期一根或稳定单脉冲认证。更优先的下一步是检查受限Hookstep如何保证至少达到已知Cauchy步的下降，而不是依据“小归一化梯度”直接转向无根或period-p结论。</p>')
sections.append('<h2>1．研究问题与三个同态对照</h2><p>问题是固定泵浦周期一求解缓慢时，还存在多大的一阶下降方向，当前半径是否过于保守，以及上行失败是否由粗矩阵差分方式主导。使用上轮27.0、27.5和27.75 mW保存末态，保持光纤模型、网格、规范和缩放不变；所有试探从同一末态出发，不累计更新。</p><p>系统仍为20.42 m、CNT → OC、OC 80%、净色散+0.2 ps²；传播窗口2.048 ns、dt=0.125 ps。内层R含光场周期闭合、EDF反转自洽、相位和时间规范，输出能量不进入代价。C输入子空间覆盖完整时间窗、±375 GHz，共6165个实正交方向。求解器步长是缩放状态范数，不是物理时间或脉冲能量。</p>'+table(['对象','改变的量','保持不变','判别量'],[['27.0/27.5 mW','沿完整merit负梯度试探','原始末态与完整输出','方向导数、实际残差下降'],['27.0/27.5 mW','六个信赖半径','同一Arnoldi基底','预测/实际下降、ρ'],['27.75 mW','粗矩阵前向→中心差分','同样h=10⁻⁶及分解策略','条件估计、独立线性残差']]))
text='<h2>2．完整输出残差在C输入子空间中的梯度</h2><p>令Φ=½‖R‖²，E把6165维扰动提升到完整状态，B=JE是扰动经过完整腔映射后的响应。计算g=BᵀR，意味着输出端保留所有残差；只有输入扰动受C子空间限制。实现于steady_support_lu.py：按批次计算中心差分列，先与完整R点乘，再进行任何输出投影。</p><p>同时记录输出投影梯度和前向差分梯度作为对照。用户建议的指标ĝ=‖g‖/(‖B‖F‖R‖)照常提供，但它受维数和奇异值分布影响，不能用固定0.01门槛判断驻点。例如J=I、R为单位向量时，ĝ=1/√d，维数超过10000就小于0.01，而负梯度仍可一步到根。</p>'
text+=table(['泵浦/mW','‖g‖','‖B‖F','ĝ','梯度响应夹角余弦','输出投影梯度相对差','前向梯度相对差'],[[r['pump_W']*1000,f"{r['gradient']['norm']:.6g}",f"{r['gradient']['frobenius_norm']:.6g}",f"{r['gradient']['normalized_frobenius']:.6g}",f"{r['gradient']['response_cosine']:.6g}",pct(r['gradient']['projected_output_relative_error']),pct(r['gradient']['forward_gradient_relative_error'])] for r in low])
text+='<p>梯度响应夹角余弦定义为‖g‖/(‖Jd_g‖‖R‖)，其中d_g=−Eg/‖g‖。它避免Frobenius维数稀释，但仍只描述这一方向；其平方等于沿该方向的线性模型可消除的merit比例。小的C子空间梯度也不等于完整JᵀR为零，更不保证二阶极小。</p>'
sections.append(text)
text='<h2>3．负梯度方向的独立导数复核与有限步试探</h2><p>梯度列采用中心h=10⁻⁶。将d_g固定后，再以10⁻⁴、10⁻⁵、10⁻⁶三种扰动尺度计算RᵀJd_g及直接merit差分，比较理论值−‖g‖。如果复核不一致，就不能把小梯度当成可靠几何证据。</p>'
text+=table(['泵浦/mW','复核h','−‖g‖','RᵀJd','直接merit导数','相对差'],[[r['pump_W']*1000,v['epsilon'],f"{-r['gradient']['norm']:.7g}",f"{v['derivative']:.7g}",f"{v['merit_derivative']:.7g}",pct(v['relative_gradient_disagreement'])] for r in low for v in r['gradient']['probes']])
for r in low:
    def draw(ax):
        line=r['gradient_line'];alpha=np.array([v['step'] for v in line if v['feasible']]);rr=np.array([v['residual']/r['physical_residual'] for v in line if v['feasible']]);ax.loglog(alpha,rr,marker='o',label='完整非线性映射');ax.axhline(1,color='gray',ls='--');ax.axvline(r['gradient']['cauchy_step'],color='orange',ls=':',label='线性模型最优步长');ax.set(xlabel='负梯度方向步长',ylabel='试探后残差 / 初始残差',title=f"{r['pump_W']*1000:g} mW负梯度方向的有限步响应");ax.legend()
    text+=figure(r['name']+'_gradient_line',draw)
    def zoom(ax):
        optimum=r['gradient']['cauchy_step'];line=[v for v in r['gradient_line'] if v['feasible'] and optimum/30<=v['step']<=2*optimum]
        ax.semilogx([v['step'] for v in line],[100*v['residual_decrease_fraction'] for v in line],marker='o');ax.axhline(0,color='gray',ls='--');ax.set(xlabel='负梯度方向步长',ylabel='完整残差下降/%',title=f"{r['pump_W']*1000:g} mW负梯度有效步长区间")
    text+=figure(r['name']+'_gradient_zoom',zoom)
    best=min((v for v in r['gradient_line'] if v['feasible']),key=lambda v:v['residual'])
    text+=f"<p>{r['pump_W']*1000:g} mW：受测负梯度步长中最好为{best['step']:.6g}，残差下降{pct(best['residual_decrease_fraction'])}；线性Cauchy步长为{r['gradient']['cauchy_step']:.6g}。这是沿一条方向的有限采样结果，不是全空间最优性证据。</p>"
sections.append(text)
text='<h2>4．固定Arnoldi基底的信赖半径扫描</h2><p>每个末态构建一次生产基准的前向粗矩阵；Krylov使用中心h=10⁻⁵，最多240维，目标线性残差0.8%。六个Hookstep共用该基底。ρ=实际merit下降/模型预测下降；另用中心h=10⁻⁶独立检查步长的模型下降，避免只依赖Hessenberg矩阵。</p>'
text+=table(['泵浦/mW','Arnoldi维数','预测线性残差','独立h=10⁻⁴','独立h=10⁻⁵','独立h=10⁻⁶'],[[r['pump_W']*1000,r['linear']['krylov'],pct(r['linear']['predicted_relative']),*[pct(v['relative']) for v in r['linear']['checks']]] for r in low])
for r in low:
    def draw(ax):
        q=r['radius_sweep'];ax.plot([v['radius'] for v in q],[100*v['residual_decrease_fraction'] if v['feasible'] else np.nan for v in q],marker='o');ax.axhline(0,color='gray',ls='--');ax.set_xscale('log');ax.set(xlabel='信赖半径Δ',ylabel='完整残差下降/%（负值为变坏）',title=f"{r['pump_W']*1000:g} mW同态信赖半径对照")
    text+=figure(r['name']+'_radius',draw)
    text+=table(['Δ','实际步长','残差下降','预测下降','实际下降','ρ','独立ρ'],[[v['radius'],f"{v['step_norm']:.6g}",pct(v['residual_decrease_fraction']),f"{v['predicted_reduction']:.5g}",f"{v['actual_reduction']:.5g}" if v['actual_reduction'] is not None else '不可行',f"{v['rho']:.5g}" if v['rho'] is not None else '—',f"{v['independent_rho']:.5g}" if v['independent_rho'] is not None else '—'] for v in r['radius_sweep']])
text+=table(['泵浦/mW','最好负梯度步长','负梯度残差下降','最好Hookstep半径','Hookstep残差下降'],[[r['pump_W']*1000,f"{min(r['gradient_line'],key=lambda v:v['residual'])['step']:.6g}",pct(max(v['residual_decrease_fraction'] for v in r['gradient_line'] if v['feasible'])),min(r['radius_sweep'],key=lambda v:v['residual'])['radius'],pct(max(v['residual_decrease_fraction'] for v in r['radius_sweep'] if v['feasible']))] for r in low])
text+='<p>两个负梯度步都远小于最小受测半径0.00156，却比受测Hookstep提供更大的实际下降。这说明当前受限Hookstep尚未达到这些已知可行步的下降量。原因可能涉及Krylov输入子空间、步长度量截断或受限求解实现；本轮没有把三者进一步分开，因此不直接断言某一实现错误。</p>'
text+='<p>较大Δ若保持正下降和较好ρ，支持该状态可接受更大步；若较大Δ变坏，只支持这个基底和状态下的局部非线性限制。两者都不能直接证明整个残差谷的形状或长期半径控制器优劣。</p>'
sections.append(text)
text='<h2>5．27.75 mW失败态的粗矩阵差分方式对照</h2><p>仅把粗矩阵列的前向差分改为中心差分，h同为10⁻⁶；完整Krylov Jv仍用中心10⁻⁵，分解阈值、截断和独立复核完全相同。中心矩阵只构建一次，未据此推进非线性迭代。条件数是投影矩阵1范数估计，不是完整物理Jacobian的条件数。</p>'
controls=[up['forward_control'],up['central_control']]
text+=table(['粗矩阵','分解方式','κ₁估计','回代误差','构建/s','Krylov维数','方向范数'],[[r['coarse']['coarse_difference'],r['coarse']['factorization'],f"{r['coarse']['coarse_condition_1_estimate']:.6g}",f"{r['coarse']['factor_probe_residual']:.6g}",f"{r['build_seconds']:.1f}",r['linear']['krylov'],f"{r['linear']['newton_norm']:.6g}"] for r in controls])
text+=table(['粗矩阵','预测残差','中心复核h=10⁻⁴','h=10⁻⁵','h=10⁻⁶'],[[r['coarse']['coarse_difference'],pct(r['linear']['predicted_relative']),*[pct(v['relative']) for v in r['linear']['checks']]] for r in controls])
def control_plot(ax):
    for r in controls:ax.loglog([v['epsilon'] for v in r['linear']['checks']],[100*v['relative'] for v in r['linear']['checks']],marker='o',label=r['coarse']['coarse_difference'])
    ax.axhline(1,color='gray',ls='--',label='1%参考线');ax.set(xlabel='独立中心差分扰动范数',ylabel='完整线性残差/%',title='27.75 mW粗矩阵差分方式与独立线性复核');ax.legend()
text+=figure('up_coarse_control',control_plot)
text+='<p>前向与中心粗矩阵均需SVD回退，条件估计及方向范数几乎相同，Krylov维数均为80；h=10⁻⁶的完整线性复核仅从2.877%变到2.849%。这一中心替换未解决失败。三个复核尺度仍给出约2.27%–7.15%，显示大方向下的导数一致性问题仍未排除；单一粗矩阵差分尺度也不足以认证真实Jacobian条件或分岔。</p>'
sections.append(text)
checks=read('step_checks.json')
sections.append('<h2>6．最佳试探步的独立重放与残差构成</h2><p>从原输入重新计算最佳负梯度步和最佳Hookstep，并按光场、EDF反转、规范三部分分解残差平方。独立重放的六个范数与原记录完全一致。负梯度步主要降低光场闭合残差，反转残差有所增加，规范项仍很小；这说明总merit下降不是各物理块同时改善。这里比较的是原方程同一缩放下的构成，不能把不同块的平方值直接当作物理误差单位。</p>'+table(['状态','试探','光场残差平方','反转残差平方','规范残差平方','复算范数差'],[[v['name'],{'initial':'初始','gradient':'负梯度','hookstep':'Hookstep'}[v['probe']],f"{v['field_squared']:.7g}",f"{v['population_squared']:.7g}",f"{v['gauge_squared']:.7g}",f"{v['replay_difference']:.3g}"] for v in checks]))
sections.append('<h2>7．结论边界与下一步判据</h2><p id="conclusion"><strong>27.0/27.5 mW仍有已验证的一阶下降方向，不支持将当前末态判为非零残差极小值。</strong>负梯度有限步分别下降约0.495%和0.925%，优于本次Hookstep半径扫描最好值0.166%和0.284%。较大半径0.025、0.05均使残差变坏，不能简单靠放大半径解决。</p><p>下一项最小算法检查应是把当前负梯度/Cauchy步纳入候选，并检查受限解是否至少达到这一步的模型下降，再区分Krylov基底缺失与步长度量截断。仅凭本轮结果就放弃period-1或直接转入period-p，证据仍不足；本轮按三项判别实验范围结束，未额外展开新的求解框架。</p><p>上一轮C优于A/B说明保留完整时间支持对受测求解有帮助，但C同时改变Fourier频率间隔，因此尚不能唯一归因于非局域时间耦合，更不能仅据此证明非正规性。27–27.5 mW的有限步末态也尚未构成稳态分支。</p><p>附件所引<a href="https://epubs.siam.org/doi/10.1137/0917003">Eisenstat与Walker的inexact Newton论文</a>讨论线性forcing term对效率及局部收敛的影响。1%线性相对误差不能单独推出10⁻³非零平台；局部理论还需要相应光滑性、非奇异性和邻域条件。本轮没有核验这些条件，也没有取得非零周期一根。</p><p>是否转入multiple shooting或period-p，应根据完整证据决定。受限梯度和六点半径扫描不足以证明局部极小；period-p还需保留动态反转状态，不能沿用周期一的静态自洽条件。</p>')
sections.append('<h2>8．复现与证据索引</h2><p>依次执行run_merit_geometry.py、check_merit_steps.py、build_merit_report.py；设置独立LASER_OUTPUT_DIR并配置CuPy。输入为公开仓库results/steady_fixed_pump_20260919三个末态及steady_tail_20260919模板。每个JSON含输入和执行源码SHA256；NPZ保存梯度、方向和受测Hookstep。合成测试覆盖完整输出梯度、中心粗矩阵导数及归一化维数反例，历史求解器默认差分设置保持兼容。</p>')
body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：残差梯度、信赖半径与粗矩阵差分对照</title><style>body{margin:0;background:#edf2f7;color:#193047;font:17px/1.8 "Microsoft YaHei",sans-serif}main{max-width:1150px;margin:auto;padding:30px 22px}section{padding:28px;background:white;border-radius:10px;margin:24px 0}h1{font-size:32px}h2{font-size:25px;color:#155785}img{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;border:1px solid #ccd8e4}.scroll{overflow:auto}</style><main><h1>原生10 MHz：残差梯度、信赖半径与粗矩阵差分对照</h1>'+''.join('<section id="s'+str(i+1)+'">'+s+'</section>' for i,s in enumerate(sections))
for name in ['down_270','down_275','up_2775','step_checks']:
    encoded=base64.b64encode((ROOT/(name+'.json')).read_bytes()).decode();body+=f'<p><a download="{name}.json" href="data:application/json;base64,{encoded}">下载{name}完整诊断记录</a></p>'
(ROOT/'原生10MHz_残差梯度与信赖半径诊断.html').write_text(body+'</main></html>',encoding='utf8')
