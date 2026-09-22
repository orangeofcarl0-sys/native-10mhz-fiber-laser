"""Self-contained Chinese report for a frozen local-geometry experiment."""
import base64,io,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':12})
d=json.loads((ROOT/'geometry.json').read_text(encoding='utf8'))
c=json.loads((ROOT/'independent_checks.json').read_text(encoding='utf8'))
s=d['stationarity'];rows=d['linear'];cur=d['curvature'];sections=[]
names={'full_gradient':'完整梯度','K64':'K64步','K96':'K96步','K192':'K192步','last_accepted':'上一接受步','tail_random_1':'高阶随机1','tail_random_2':'高阶随机2','tail_random_3':'高阶随机3'}
def table(head,body):
    return '<div class="table"><table><tr>'+''.join('<th>'+str(x)+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in body)+'</table></div>'
def section(title,body,draw=None):
    pic=''
    if draw:
        fig,ax=plt.subplots(figsize=(11,5.8));draw(ax);fig.tight_layout();b=io.BytesIO();fig.savefig(b,format='png',dpi=160);fig.savefig(ROOT/f'figure_{len(sections)+1:02d}.png',dpi=160);plt.close(fig)
        pic='<img alt="'+title+'" src="data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()+'">'
    sections.append('<section id="s'+str(len(sections)+1)+'"><h2>'+title+'</h2>'+body+pic+'</section>')
section('研究问题与本轮结论边界',
 '<p>目标是解释固定参数周期一求根为何缓慢：线性空间尚未充分展开、GN遗漏曲率、single-shooting方程形式，还是非零驻点。前三种机制可能同时存在，不能按互斥假设机械分流。</p>'
 '<p><b>本轮直接测到：384维线性问题尚未达到完整法方程收敛；关键步方向存在约25%–35%的正遗漏曲率。</b>两者同时成立。目前既不能认定“只需更深GKB”，也不能认定“固定参数下无根”。</p>'
 '<p>仅冻结诊断，不执行无约束线性最优步，不增加outer轨迹，不修改生产求解器。下文区分实测事实、条件判断和仍未回答的问题。</p>')
section('冻结末态、物理参数与状态度量',
 f'<p>起点来自上一轮fresh192臂第20接受步，R={d["protocol"]["initial_residual"]:.12g}。不是pure64末态，也不是recycled臂末态。泵浦27.5 mW、输出耦合80%、净色散+0.2 ps²、CNT饱和功率40 W、CNT→OC、总腔长20.42 m、10 MHz全部保留。</p>'
 '<p>网格16384点、dt=0.125 ps、窗口2.048 ns；实状态65553维。沿用光场归一化、反转比例和相位/时间gauge。所有范数与χ都依赖这个既定状态/残差度量，不与另一种缩放下的数值直接比较。EDF反转仍是联合求解变量。</p>'
 '<p>Φ=½‖R‖²；g=JᵀR；HΦ=JᵀJ+ΣRᵢ∇²Rᵢ。Jv来自归一化方向中心差分，Jᵀv来自已经验证的离散伴随。</p>')
section('一阶驻点指标的数值与含义',table(['量','结果'],[
 ['‖R‖',f'{d["protocol"]["initial_residual"]:.10g}'],['‖JᵀR‖',f'{s["gradient_norm"]:.10g}'],
 ['σmax估计',f'{s["sigma_estimate"]:.10g}'],['χ=‖JᵀR‖/(σmax‖R‖)',f'{s["chi"]:.10g}'],['四个谱范数估计的相对跨度',f'{s["sigma_estimate_spread"]:.3g}']])+
 '<p>χ约0.00263，表示在当前度量下梯度相对较小；它不是驻点证明，也不是R在Range(J)内的投影比例。小奇异值方向也会压低χ，即使线性方程完全有解。</p>')
def power(ax):
    for p in d['power']:ax.semilogy([t['iteration'] for t in p['history']],[t['eigen_residual'] for t in p['history']],label=f'独立起点{p["start"]+1}')
    ax.axhline(1e-5,color='gray',ls='--',label='本轮停止门');ax.set(xlabel='幂迭代次数',ylabel='相对特征对残差');ax.legend();ax.grid(alpha=.2)
section('最大奇异值估计的收敛检查',
 '<p>三个独立随机起点运行JᵀJ幂迭代，至少20次，必要时最多60次；均在20次达到相对特征残差&lt;10^-5。再与B384最大奇异值交叉比较。四者一致支持此谱范数估计可靠，但不构成严格全局上界或完整谱认证。</p>',power)
def eta(ax):
    ax.plot([r['k'] for r in rows],[r['eta'] for r in rows],'o-',label='实际保存JV响应');ax.set(xlabel='GKB维数k',ylabel='ηlin = ‖R + Js‖ / ‖R‖',ylim=(.99,1.001));ax.grid(alpha=.2);ax.legend()
section('深层线性最小二乘残差随维数变化',
 '<p>在相同冻结状态构造GKB至384维；每个节点以SVD解无约束reduced LS，截断为最大奇异值的10^-12。η使用实际保存的JV响应重算，不只读取名义双对角矩阵。纵轴刻意放大接近1的范围，不能误读成残差已接近零。</p>'
 f'<p>K192 η={rows[3]["eta"]:.7f}；K384 η={rows[-1]["eta"]:.7f}。K384只在线性模型中消除约{100*(1-rows[-1]["eta"]):.3f}%残差范数，或{100*(1-rows[-1]["eta"]**2):.3f}%原始merit。不能把这些未执行步当作真实腔映射的推进。</p>',eta)
def normal(ax):
    ax.semilogy([r['k'] for r in rows],[r['full_normal_ratio'] for r in rows],'o-',label='完整法方程残差 / 初始梯度')
    ax.semilogy([r['k'] for r in rows],[r['projected_normal_ratio'] for r in rows],'s--',label='V内投影法方程残差 / 初始梯度')
    ax.set(xlabel='GKB维数k',ylabel='相对法方程残差');ax.legend();ax.grid(alpha=.2)
section('子空间内最优与完整线性问题收敛的区别',
 f'<p>K384的投影法方程残差为{rows[-1]["projected_normal_ratio"]:.3g}，但完整‖Jᵀ(R+Js)‖/‖g‖仍为{rows[-1]["full_normal_ratio"]:.4f}。这说明所建空间内已精确求解，空间外仍有显著一阶信息。完整法方程残差不要求随LSQR维数单调下降。</p>'
 '<p><b>因此η≈0.993不能解释为已确定的range外分量或不可消除残差下限。</b>384/65553维仍是有限线性诊断，不是全空间求解证明；没有获得可用于判断秩亏或无根的线性残差平台。</p>',normal)
section('线性步范数、奇异值与累计成本',table(['k','ηlin','完整normal比','线性步范数','σmin(Bk)','κ₂(Bk)','构基累计/s'],[[r['k'],f'{r["eta"]:.7f}',f'{r["full_normal_ratio"]:.4f}',f'{r["step_norm"]:.6g}',f'{r["singular_values"][-1]:.4g}',f'{r["singular_values"][0]/r["singular_values"][-1]:.1f}',f'{r["seconds"]:.2f}'] for r in rows])+
 '<p>这里没有信赖半径约束，也没有把K384约0.0362的步直接执行。随着空间加深，捕获到更弱方向，但B384条件数不是完整J的条件数；不能由它排除全系统的弱奇异方向。累计时间只指构基，不含后续SVD/伴随核验和文件压缩。</p>')
def spectrum(ax):
    for r in rows:ax.semilogy(np.arange(1,len(r['singular_values'])+1),r['singular_values'],label='K'+str(r['k']))
    ax.set(xlabel='按降序排列的奇异值编号',ylabel='σ(Bk)');ax.legend(ncol=2);ax.grid(alpha=.2)
section('各深度reduced矩阵的奇异谱',
 '<p>谱图只展示已捕获的空间。最小奇异值随扩维继续降低，与弱方向尚在展开一致；不做全空间谱底、秩或根存在性推断。</p>',spectrum)
section('完整Hessian与GN曲率的测量方法',
 '<p>对单位方向v，用HΦv≈[g(x+hv)−g(x−hv)]/(2h)，并单独计算JᵀJv。记录qGN=‖Jv‖²、qfull=vᵀHΦv、qmissing=qfull−qGN以及κ=|qmissing|/qGN。</p>'
 '<p>方向为当前−g、当前K64/K96/K192无约束步、上一接受步（在当前末态评估），以及GKB第321–384列的三个固定随机组合。后者是“高阶列组合”，不是特意寻找的最弱奇异向量，也不是全方向曲率抽样认证。</p>'
 '<p>五个尺度h=10^-4、3×10^-5、10^-5、3×10^-6、10^-6；10^-5作参考，两个更细尺度相对qGN变化均&lt;1%才认可该方向分类。另用独立物理映射的标量二阶差分复核。</p>')
def missing(ax):
    y=[r['scales'][2]['missing']/r['scales'][2]['gn'] for r in cur];ax.bar(np.arange(len(y)),y,color=['#176c91' if t>=0 else '#c05c39' for t in y]);ax.axhline(0,color='black',lw=.7)
    ax.set_xticks(np.arange(len(y)),[names[r['name']] for r in cur],rotation=20);ax.set(ylabel='带符号遗漏曲率 / GN曲率');ax.grid(axis='y',alpha=.2)
section('关键步方向的遗漏曲率大小与正负号',
 '<p>K64/K96/K192和上一接受步上，遗漏项分别约+24.6%、+35.0%、+32.7%、+34.4%。这些是正曲率：GN低估了这些方向的局部二阶弯曲，具有过度预测下降的可能。梯度方向和三个随机高阶组合上遗漏项很小。</p>'
 '<p>这是方向依赖的中等曲率修正，既不是“GN完全准确”，也不是“GN模型已被证明全面失效”。所有被测方向qfull为正；不能据此证明完整Hessian正定，未搜索最负曲率方向。</p>',missing)
section('各方向曲率数值与多尺度稳定性',table(['方向','qGN','qmissing（带符号）','qfull','κ','细尺度最大变化/qGN','通过'],[[names[r['name']],*[f'{r["scales"][2][key]:.6g}' for key in ['gn','missing','full','kappa']],f'{max(r["fine_scale_errors"]):.3g}',r['validated']] for r in cur]))
def convergence(ax):
    for row in cur[:5]:ax.semilogx([t['h'] for t in row['scales']],[t['missing']/t['gn'] for t in row['scales']],'o-',label=names[row['name']])
    ax.set(xlabel='中心差分尺度h',ylabel='带符号遗漏曲率 / GN曲率');ax.legend();ax.grid(alpha=.2)
section('Hessian方向二阶量的差分尺度检查',
 '<p>图中不同h曲线稳定支持测到的25%–35%修正来自局部曲率，而非选取某一个差分尺度的偶然误差。梯度方向近零，不能用它代替实际求解步方向的诊断。</p>',convergence)
section('局部曲率与有限步长ρ的关系',
 '<p>qfull−qGN是零点附近的二阶遗漏；上一轮ρ则来自有限步长完整腔映射，二者不能直接画等号。在本轮关键步上正遗漏项可解释部分模型乐观倾向，但不能把κ≈0.33直接转换成ρ≈0.67。</p>'
 '<p>Φ(x+s)的局部展开还包含高阶余项。只有在同一方向、足够小的幅度下，−½sᵀ(HΦ−JᵀJ)s才描述actual−prediction的主导二阶项。本轮没有新增长步或outer接受点。</p>')
section('独立物理映射复核与数值健康',table(['检查','结果'],[
 ['方向数 / 线性节点数 / Hessian交叉对称性对数',f'{len(c["curvature"])} / {len(c["linear"])} / {len(c["symmetry"])}'],
 ['独立二阶差分最佳联合误差的方向最大值/qGN',f'{max(t["best_joint_error"] for t in c["curvature"]):.3g}'],
 ['Hessian交叉对称性最大operator-scaled误差',f'{max(t["operator_scaled_error"] for t in c["symmetry"]):.3g}'],
 ['三尺度线性响应复核最大相对误差',f'{max(max(t["three_scale_response_errors"]) for t in c["linear"]):.3g}'],
 *[[key,f'{val:.4g}'] for key,val in d['health'].items()]])+
 '<p>独立检查直接调用物理腔映射，而非仅重复伴随primal；检查Φ标量二阶差分和R加权残差二阶导。标量差分在极细h有消减误差，因此完整保留四尺度结果，表中报告每方向最佳尺度的联合误差而非伪称全部尺度同精度。对称误差以两个HVP范数之和归一化。</p>'
 '<p>两个已知答案测试涵盖负完整曲率和“小χ但线性方程可解”的反例。数值健康只能验证诊断实现，不能代替根存在性、稳定性或实验认证。</p>')
section('小χ不等于线性不可解：一个已验证反例',
 '<p>取J=diag(1,10^-6)，R=(0,1)。则χ=10^-6，但J满秩，s=(0,−10^6)使R+Js=0，η=0。小χ在这里反映弱灵敏度，而不是range外残差。</p>'
 '<p>同理，即使当前精确满足JᵀR=0且R≠0，也只证明局部驻点；仍需区分极小、鞍点或极大，更不能推出同参数下其他位置不存在周期根。这是本轮解释χ和η必须保留的边界。</p>')
gate=d['pilot_decision']
section('小型multiple-shooting对照的预设触发条件',table(['冻结的操作性条件','本轮证据','触发'],[
 ['χ≥0.01、η384≤0.1且至少一方向验证κ≥0.5',f'χ={s["chi"]:.5g}；η={rows[-1]["eta"]:.5g}；最大κ={max(t["scales"][2]["kappa"] for t in cur):.5g}',gate['curved_solvable']],
 ['η≥0.5、完整normal比<0.001且末两次η相对改善<0.001',f'完整normal比={rows[-1]["full_normal_ratio"]:.5g}；末两次改善={gate["last_eta_marginals"]}',gate['resolved_high_floor']]])+
 '<p>这些是执行前设定的实验分流条件，不是物理常数或数学定理。本轮均未满足，因此没有追加m=2/4 pilot，没有将主线切到multiple shooting。它仍是待验证的formulation假设，不能由本轮数据支持或排除。</p>')
section('四类机制的证据与尚未回答的问题',table(['机制','本轮能说什么','本轮不能说什么'],[
 ['H1：线性信息未充分展开','完整法方程残差未小；支持有限Krylov空间尚未解决线性问题','不能因此证明根存在，或决定直接用K384长跑'],
 ['H2：GN遗漏曲率','关键步正遗漏约25%–35%，跨尺度与独立映射复核一致','不能宣布它是唯一瓶颈或full Newton必然更快'],
 ['H3：single-shooting formulation','本轮没有满足预设pilot门，尚无配对formulation实验','不能用η≈1或有限步ρ单独归因于formulation'],
 ['H4：非零驻点／固定参数无根','χ约0.00263，提示需关注弱方向或近驻点','未达到JᵀR=0认证；没有局部极小或全局无根证据']])+
 '<p><b>本轮结论是“线性分辨率不足与方向依赖的曲率修正并存”，而不是四选一已完成。</b>先保留这些冻结证据；不自动增加生产深度、不调控制器，也不把H3/H4未解当作已经否定。</p>')
section('结果文件与可复现范围',
 '<p>公开vectors.npz包含冻结x/R/g、七个线性步及残差、八个方向/Jv/GN-HVP/参考完整HVP和B384。geometry.json保存谱、全部尺度标量与分流结果；independent_checks.json保存独立核验；protocol.json含起点哈希。</p>'
 f'<p>完整V/JV保存在本地local_basis.npz，SHA-256：<code>{d["local_basis_sha256"]}</code>。五尺度HVP保存在本地local_hessians.npz。大基不重复上传，但run_local_geometry.py可由已公开上一轮末态重建。</p>'
 f'<p>测量阶段耗时约{d["seconds"]:.1f}s（不含最终压缩归档）；GKB构基约{d["basis_seconds"]:.1f}s。同一GPU顺序执行，没有并发数值作业。测量是确定性单次审计，不将计时写成统计置信区间。</p>'
 '<p>本报告不覆盖网格再认证、Floquet稳定性或自启动性；当前R约9.32×10^-4的状态仍不是已认证稳定单脉冲解。</p>')
style='body{margin:0;background:#edf2f6;color:#182d40;font:17px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:25px}section{background:white;border-top:4px solid #176c91;padding:28px;margin:25px 0}h1{font-size:30px}h2{font-size:24px;margin-top:0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cddbe4;padding:8px;text-align:left}th{background:#e8f1f6}.table{overflow:auto}code{overflow-wrap:anywhere}a{color:#176c91}@media(max-width:600px){main{padding:10px}section{padding:14px}h1{font-size:24px}h2{font-size:20px}}'
payload=base64.b64encode(json.dumps(dict(geometry=d,checks=c),ensure_ascii=False).encode()).decode()
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：周期一局部几何诊断</title><style>'+style+'</style><main><h1>原生10MHz：周期一局部几何诊断</h1>'+''.join(sections)+f'<a download="local_geometry.json" href="data:application/json;base64,{payload}">下载完整诊断记录</a></main></html>'
(ROOT/'原生10MHz_周期一局部几何诊断.html').write_text(html,encoding='utf8')
print('REPORT',len(sections))
