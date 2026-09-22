"""Standalone Chinese report: saved-space trust maps and ray diagnostics."""
import base64,io,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':12})
a=json.loads((ROOT/'radius_map.json').read_text(encoding='utf8'));c=json.loads((ROOT/'independent_checks.json').read_text(encoding='utf8'))
rows=a['rows'];lookup={r['name']:r for r in rows};checks={r['name']:r for r in c['checks']};sections=[]
radii=a['protocol']['radii'];depths=a['protocol']['depths']
def table(head,data):
    return '<div class="table"><table><tr>'+''.join('<th>'+str(v)+'</th>' for v in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in data)+'</table></div>'
def section(title,body,draw=None):
    picture=''
    if draw:
        fig,ax=plt.subplots(figsize=(11,6));draw(ax);fig.tight_layout();b=io.BytesIO();fig.savefig(b,format='png',dpi=150)
        fig.savefig(ROOT/f'figure_{len(sections)+1:02d}.png',dpi=150);plt.close(fig)
        picture='<img alt="'+title+'" src="data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()+'">'
    sections.append('<section id="s'+str(len(sections)+1)+'"><h2>'+title+'</h2>'+body+picture+'</section>')
def heat(ax,key,scale=1):
    values=np.array([[lookup[f'K{k}_D{i}'][key]*scale for i in range(5)] for k in depths])
    im=ax.imshow(values,aspect='auto',cmap='coolwarm');plt.colorbar(im,ax=ax)
    ax.set(xticks=range(5),xticklabels=radii,yticks=range(9),yticklabels=depths,xlabel='信赖半径 Δ（缩放状态范数）',ylabel='GKB维数 k')
    for j in range(9):
        for i in range(5):ax.text(i,j,f'{values[j,i]:.3f}',ha='center',va='center',fontsize=10)
section('冻结状态、实验问题与比较口径',
 '<p>本轮复用上一轮保存的V₁₉₂、JV、B和旧64维基在当前状态的响应；<b>没有新Jv/J^Tv调用、没有新outer轨迹</b>。初始R=9.746756121×10⁻⁴，泵浦27.5 mW、OC=80%、GDD=+0.2 ps²、Psat=40 W、CNT→OC、总腔长20.42 m、10 MHz、网格和gauge全部不变。</p>'
 '<p>45点fresh地图（9维数×5半径），5点回收联合空间，15点固定方向ray。研究问题：缩小半径并重新求解，能否把深层空间的线性收益转化为真实下降？这与沿原候选步缩放是两个独立对照。</p>'
 '<p>步长和半径使用既有缩放状态的欧氏度量，不是米或皮秒。相同半径不一定意味着相同步长：无约束解已在半径内时，候选不会改变。ray用绝对范数实现真正等步长比较。</p>')
section('模型下降、真实下降与非线性缺陷的定义',
 '<p>Φ(x)=½‖R(x)‖²；Δm=−R^TJs−½‖Js‖²；ΔΦ=Φ(x)−Φ(x+s)；ρ=ΔΦ/Δm。Δm是冻结Jacobian预测，ΔΦ由完整腔映射计算。正值表示下降，负值表示恶化。</p>'
 '<p>每个半径重新SVD求解 min‖y‖≤Δ ½‖βe₁−Bₖy‖²，再取s=Vₖy。λ&gt;0表示边界约束生效；λ=0表示空间内无约束最小二乘步。回收空间保持相同状态范数，正交合并后重新解小问题。</p>'
 '<p>e=R(x+s)−R(x)−Js；εNL=‖e‖/‖Js‖。Js直接由保存的JV乘以系数得到。εNL测量残差向量的有限步长误差；ρ只衡量目标函数下降预测，两者不能互相替代。</p>')
section('不同深度与半径的真实目标下降',
 '<p>图中数值单位为10^(-9)。0.025下深层空间出现负下降，0.0125下恢复正收益；K64所有半径结果相同，因为原步范数约0.001195，已经小于最小测试半径。</p>',lambda ax:heat(ax,'actual',1e9))
section('不同深度与半径的模型兑现比例ρ',
 '<p>ρ&gt;0.1是既有验收线；接近1表示目标下降预测接近实际，不代表残差向量本身线性化误差很小。所有候选可行性与Cauchy下界仍记录，失败候选保留显示。</p>',lambda ax:heat(ax,'rho'))
section('不同深度与半径的非线性缺陷εNL',
 '<p>εNL随步增大而明显增加。K192、Δ=0.0125时εNL≈0.823，即使ρ≈1.009，向量误差仍达线性响应范数的82%。下文通过merit恒等式解释其原因，不能把该点称作严格线性区。</p>',lambda ax:heat(ax,'nonlinear_defect'))
def actual_by_radius(ax):
    for k in [64,96,128,192]:ax.plot(radii,[lookup[f'K{k}_D{i}']['actual']*1e9 for i in range(5)],'o-',label=f'K{k}')
    ax.plot(radii,[lookup[f'UNION_D{i}']['actual']*1e9 for i in range(5)],'s--',label='K64+R64')
    ax.axhline(0,color='gray');ax.set(xlabel='信赖半径 Δ',ylabel='真实ΔΦ / 10^(-9)');ax.legend();ax.grid(alpha=.2)
section('fresh与回收空间的同半径对照',
 '<p>回收空间在0.0125取得ΔΦ=4.69009×10^(-9)、ρ=1.12886。它达到同半径K192的102.1%，也接近本次fresh最优K176的100.15%。这支持继续研究回收，但0.15%的单态差异不足以宣称普遍优于fresh。</p>'
 '<p>沿用上一轮独立构造计时：回收warm约52.4 s，fresh192约134.9 s；这是基缓存可用时的历史成本参考。本轮复用已有基，不重新测量生产吞吐率。</p>',actual_by_radius)
def norm_lambda(ax):
    for k in [64,96,128,192]:ax.plot(radii,[lookup[f'K{k}_D{i}']['step_norm'] for i in range(5)],'o-',label=f'K{k}')
    ax.plot(radii,radii,'k--',label='‖s‖=Δ');ax.set(xlabel='半径Δ',ylabel='实际步范数');ax.legend();ax.grid(alpha=.2)
section('实际步范数与trust multiplier',
 '<p>相同半径的比较是相同可用预算；图中的水平段说明无约束步已经不再变化。λ随半径的记录用于区分边界解和无约束解。</p>'+table(['空间','Δ=.003125','Δ=.00625','Δ=.0125','Δ=.01875','Δ=.025'],[[f'K{k}']+[f'{lookup[f"K{k}_D{i}"]["lambda_value"]:.3g}' for i in range(5)] for k in [64,96,128,192]]),norm_lambda)
for k in [96,128,192]:
    def draw(ax,k=k):
        data=sorted([r for r in rows if r['kind']=='ray' and r['k']==k],key=lambda r:r['step_norm'])
        ax.plot([r['step_norm'] for r in data],[r['prediction']*1e9 for r in data],'o--',label='沿原方向的线性预测')
        ax.plot([r['step_norm'] for r in data],[r['actual']*1e9 for r in data],'s-',label='沿原方向的真实下降')
        trust=[lookup[f'K{k}_D{i}'] for i in range(5)]
        ax.scatter([r['step_norm'] for r in trust],[r['actual']*1e9 for r in trust],marker='x',s=65,label='重新解trust的真实下降')
        ax.axvline(lookup[f'K{k}_D4']['step_norm'],color='gray',ls=':',label='原始候选范数')
        ax.axhline(0,color='gray');ax.set(xlabel='绝对步范数 ‖s‖',ylabel='目标下降 / 10^(-9)');ax.legend();ax.grid(alpha=.2)
    section(f'K{k}固定方向的步长扫描与重新求解对照',
      '<p>ray的方向固定，只有α改变；×号则是减小半径后重新求解得到的候选，方向可以改变。折线仅连接离散采样点，不代表已定位连续最优步长。K96中0.0125与0.01875超过原步长，属于α&gt;1外推。</p>',draw)
def defects(ax):
    for k in [96,128,192]:
        data=sorted([r for r in rows if r['kind']=='ray' and r['k']==k],key=lambda r:r['step_norm'])
        ax.plot([r['step_norm'] for r in data],[r['nonlinear_defect'] for r in data],'o-',label=f'K{k}')
    ax.set(xlabel='绝对步范数 ‖s‖',ylabel='εNL=‖有限步长缺陷‖/‖Js‖');ax.legend();ax.grid(alpha=.2)
section('等步长下不同候选方向的非线性缺陷',
 '<p>深层方向在相同范数下具有更大的向量非线性缺陷，但这不等于更差的真实下降。0.0125时K192 ray真实下降仍高于K96 ray，说明“晚期方向非线性更强”与“它们仍有用”可以同时成立。</p>',defects)
def decomposition(ax):
    names=['K96_D4','K192_D2','K192_D4','UNION_D2'];idx=np.arange(len(names))
    ax.bar(idx-.2,[checks[n]['cross_term']*1e9 for n in names],width=.4,label='−(R+Js)^Te')
    ax.bar(idx+.2,[checks[n]['defect_penalty']*1e9 for n in names],width=.4,label='−½‖e‖²')
    ax.set(xticks=idx,xticklabels=['K96原步','K192 Δ=.0125','K192 Δ=.025','回收 Δ=.0125'],ylabel='实际−线性预测的分项 / 10^(-9)');ax.axhline(0,color='gray');ax.legend()
section('ρ接近1与较大非线性缺陷为何可以同时出现',
 '<p>精确恒等式：ΔΦ−Δm=−(R+Js)^Te−½‖e‖²。交叉项可正可负，而缺陷平方项总为负；两项抵消时，即使εNL较大，ρ仍可接近1。因此ρ是可靠的merit验收量，却不是“整个残差已线性化”的证明。</p>'
 '<p>本轮将缺陷分为field、population、gauge三块归档；这只能定位残差层次，尚不能单独归因于CNT、EDF或某一传播段。</p>',decomposition)
section('完整候选表与数值核验',
 f'<p>65点独立重放，最大残差差异={c["max_residual_difference"]:.3g}；merit缺陷恒等式最大误差={c["max_identity_error"]:.3g}。原半径候选复现上一轮；所有fresh reduced解检查SVD KKT、步长约束；保存JV的线性预测与B预测交叉核对。没有重新计算导数。</p>'
 '<p>ray是机制诊断；表中的通过标记不表示它已被生产求解器接受，也不代表执行了新的outer步。</p>'
 +'<details><summary>展开全部65点（含λ、外推标记）</summary>'+table(['候选','范数','Δm','ΔΦ','ρ','εNL','λ','外推','保护'],[[r['name'],f'{r["step_norm"]:.6g}',f'{r["prediction"]:.6g}',f'{r["actual"]:.6g}',f'{r["rho"]:.4f}',f'{r["nonlinear_defect"]:.4f}',str(r['lambda_value']),r.get('extrapolated',False),r['passed']] for r in rows])+'</details>')
section('本冻结状态支持的生产化方向与证据边界',
 '<p><b>优先保留深层GKB信息，配合半径内层重解；回收空间同样值得进入下一轮受控比较。</b>K192在Δ=0.0125恢复ρ=1.009，实际下降比原K96高34.2%，不支持仅因原半径失败就限制深度为96。</p>'
 '<p>回收空间在该半径与fresh深层空间实际下降接近，旧计时也显示缓存可用时有成本优势。因此“深层/回收空间 + 内层半径控制”均获支持，二者并非互斥选项。不能仅凭单个冻结状态确定长程最优算法。</p>'
 '<p>0.0125是本次离散半径网格中的最佳区间，不是已证明的全局最优半径。192维也不是已证明的最佳深度；fresh网格最佳为176维，和回收结果差异很小。下一轮可比较受控outer轨迹，但本轮未执行，不切multiple shooting，也不引入预条件器。</p>')
style='body{margin:0;background:#edf2f6;color:#182d40;font:17px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:25px}section{background:white;border-top:4px solid #176c91;padding:28px;margin:25px 0}h1{font-size:30px}h2{font-size:24px;margin-top:0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cddbe4;padding:8px;text-align:left}th{background:#e8f1f6}.table{overflow:auto}a{color:#176c91}@media(max-width:600px){main{padding:10px}section{padding:14px}h1{font-size:24px}h2{font-size:20px}}'
payload=base64.b64encode(json.dumps(dict(result=a,checks=c),ensure_ascii=False).encode()).decode()
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：GKB非线性半径地图</title><style>'+style+'</style><main><h1>原生10MHz：固定GKB空间的非线性半径地图</h1>'+''.join(sections)+f'<a download="radius_map.json" href="data:application/json;base64,{payload}">下载完整记录</a></main></html>'
(ROOT/'原生10MHz_GKB固定空间非线性半径地图.html').write_text(html,encoding='utf8');print('REPORT',len(sections))
