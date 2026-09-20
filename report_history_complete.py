"""Offline evidence report for full-bank, equal-cardinality and novelty controls."""
import base64,io,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT,PROJECT
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':12})
a=json.loads((ROOT/'history_complete.json').read_text());old=json.loads((PROJECT/'results/steady_active_history_20260920/active_history.json').read_text())
checks=json.loads((ROOT/'independent_checks.json').read_text());parts=[]
def find(g,d):return next(v for v in a['trials']+old['trials'] if v['group']==g and v['radius']==d)
def table(head,rows):return '<div class="table"><table><tr>'+''.join('<th>'+h+'</th>' for h in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
def section(title,text,draw=None):
    pic=''
    if draw:
        fig,ax=plt.subplots(figsize=(11,5.6));draw(ax);fig.tight_layout();b=io.BytesIO();fig.savefig(b,format='png',dpi=150)
        fig.savefig(ROOT/f'figure_{len(parts)+1:02d}.png',dpi=150);plt.close(fig)
        pic='<img alt="'+title+'" src="data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()+'">'
    parts.append('<section><h2>'+title+'</h2>'+text+pic+'</section>')
section('研究问题与本轮固定条件',
    '<p>本轮补齐 7 个历史方向和环带方向的联合比较，回答：完整历史库能否替代环带、贪心选择是否遗漏组合、环带在现有输入与响应空间之外还有多少分量。'
    '只使用同一个 S3 末态，‖R‖='+f"{a['initial_residual']:.12g}"+'；没有续算或改变物理参数。</p>'
    '<p>20.42 m、10 MHz、CNT→OC、OC=80%、净色散 +0.2 ps²、泵浦 27.5 mW、Psat=40 W；归一化、网格和规范固定。'
    '上一轮未存完整 Arnoldi 基底，本轮重建共同 C/Arnoldi，并复核上一轮全部 15 个模型候选。'
    '环带方向复用已存结果，不重扫 6144 个方向；所有方向响应在当前状态重新计算。</p>'
    '<p>Gk：贪心加入 k 个历史方向；Gk+A：同一组合加环带；E4：穷举得到的最佳四历史方向。每组都含共同 Arnoldi 基底和当前 C 方向。'
    '比较三个信赖域半径；merit 为 Φ=‖R‖²/2，实际下降为 Φ(x)−Φ(x+s)。求解步不代表物理时间。</p>')

ratio_rows=[]
for d in a['radii']:
    g7=find('G7',d);ga=find('G7+A',d);e4=find('E4',d);g4=find('G4',d)
    ratio_rows.append([d,f"{ga['prediction']/g7['prediction']:.4f}",f"{ga['actual']/g7['actual']:.4f}",f"{g4['prediction']/e4['prediction']:.4f}",f"{g4['actual']/e4['actual']:.4f}"])
section('完整历史库与环带的同半径增益',
    '<p><b>本轮观察：G7+A 的实际下降仍为 G7 的约 2.52–2.57 倍，完整历史库没有替代当前环带方向。</b>'
    '在全部三个半径、每一种历史数量下，贪心均达到 128 子集枚举的相同数量最佳预测值；四方向 E4 与 G4 的真实下降也一致。'
    '所以本轮没有发现贪心组合选择损失，仍有收益主要来自增加方向和环带信息。</p>'
    '<p>前两列比较 G7+A/G7：大于 1 表示环带在完整历史库基础上仍有收益；1.25 是本轮参考增益标准。'
    '后两列比较 G4/E4：衡量贪心在相同四方向预算下是否接近最优。模型预测与完整映射实际下降分别列出，不能相互替代。</p>'
    +table(['半径','G7+A/G7 预测比','G7+A/G7 实际比','G4/E4 预测比','G4/E4 实际比'],ratio_rows))

def curve(ax):
    d=.003125
    for suffix,label in [('', '仅历史'),('+A','历史 + 环带')]:
        ks=range(4,8);ax.plot(list(ks),[find(f'G{k}{suffix}',d)['actual'] for k in ks],'o-',label=label)
    ax.set(xlabel='贪心历史方向数',ylabel='完整映射实际 merit 下降',xticks=range(4,8));ax.legend();ax.grid(alpha=.2)
section('增加到七个历史方向后的实际下降',
    '<p>图中固定 Δ=0.003125；全部半径的数据见下表。增加子空间自由度不会降低精确最优模型收益，'
    '但真实非线性下降不保证单调，尤其候选越来越接近信赖域边界时。</p>'
    +table(['半径','方法','预测下降','实际下降','ρ','通过'],[[r['radius'],r['group'],f"{r['prediction']:.7g}",f"{r['actual']:.7g}",f"{r['rho']:.5f}",'是' if r['passed'] else '否'] for r in a['trials']]),curve)

def exhaustive(ax):
    for audit in a['exhaustive']:
        g=next(v for v in a['greedy'] if v['radius']==audit['radius'])
        ratios=[g['ladder'][k-1]['prediction']/audit['best_by_count'][k]['prediction'] for k in range(1,8)]
        ax.plot(range(1,8),ratios,'o-',label=f"Δ={audit['radius']}")
    ax.axhline(.95,color='gray',ls='--');ax.set(xlabel='相同历史方向数量',ylabel='贪心 / 穷举最佳预测下降',xticks=range(1,8));ax.legend();ax.grid(alpha=.2)
subset_rows=[]
for audit in a['exhaustive']:
    for k,b in enumerate(audit['best_by_count'][1:],1):
        subset_rows.append([audit['radius'],k,', '.join(str(a['history_outer_steps'][i]) for i in b['selected']),f"{b['prediction']:.7g}"])
section('128 个历史子集的等数量穷举对照',
    '<p>每个半径枚举 128 个子集，按方向数分别找最大预测下降。不限制方向数时完整七方向空间天然包含全部子集，'
    '不能拿其胜出来证明贪心有效。这里用相同数量的最优子集做分母，E4 另经完整映射验证。其余最佳子集仅有模型证据。</p>'
    +table(['半径','方向数','最佳来源外迭代编号','最大预测下降'],subset_rows),exhaustive)

def novelty_plot(ax):
    n=a['novelty']['1e-12'];ax.bar(['状态空间外 νx','响应空间外 νR','同系数联合误差'],[n['state']['relative'],n['response']['relative'],n['joint']['relative']])
    ax.set(ylabel='归一化剩余范数',ylim=(0,1.1));ax.grid(axis='y',alpha=.2)
section('环带方向在完整历史模型之外的分量',
    '<p>输入空间 S=span(Z,dC,D₇)，响应空间 Y=span(VH,JdC,JD₇)。νx=‖dA−PₛdA‖/‖dA‖，'
    'νR=‖JdA−PᵧJdA‖/‖JdA‖。使用 SVD 和相对奇异值容差；Y 不包含人为追加的残差目标列。'
    '投影使用已验证等价的正交压缩坐标。</p>'
    '<p>两项分别拟合可能使用不同系数，因此补充同一组系数拟合输入和响应的联合诊断。联合上下块分别除以 ‖dA‖、‖JdA‖；'
    '这是解释性诊断，不改变求解器的信赖域尺度或物理方程。两个独立投影误差小，仍不能保证成对表示准确。</p>'
    '<p><b>当前 νx≈0.98856、νR≈0.97408，联合误差≈0.98526。</b>三个容差结果一致，均保留 46 维。'
    '因此环带方向及其响应大部分位于本轮模型之外；不是只靠在既有模型中换一组系数即可精确替代。'
    '百分比是范数比例，不是光功率比例，也不是最终求解步的环带占比。</p>'
    +table(['SVD 容差','νx','νR','联合误差','输入秩','响应秩'],[[t,f"{n['state']['relative']:.7g}",f"{n['response']['relative']:.7g}",f"{n['joint']['relative']:.7g}",n['state']['rank'],n['response']['rank']] for t,n in a['novelty'].items()]),novelty_plot)

def marginal(ax):
    for g in a['greedy']:ax.plot(range(1,8),[100*v['marginal_gain'] for v in g['ladder']],'o-',label=f"Δ={g['radius']}")
    ax.axhline(5,color='gray',ls='--');ax.set(xlabel='新增第几个历史方向',ylabel='相对已有预测下降的增量 (%)',xticks=range(1,8));ax.legend();ax.grid(alpha=.2)
section('边际收益与提前停止规则的适用范围',
    '<p>继续记录完整七级边际收益，不应用“首次低于 5% 就停止”的策略。'
    '组合目标未表现出可直接假定的递减收益性质，早停可能漏掉后续互补方向。'
    '当前只有七个历史方向，全部纳入是可直接比较的基线；容量淘汰和事件触发机制本轮尚未实施。</p>',marginal)

maxerror=max(c['discrepancy'] for r in a['trials'] for c in r['checks'])
section('数值复核与后续算法决策边界',
    f"<p>{sum(r['passed'] for r in a['trials'])}/{len(a['trials'])} 个新候选通过 Cauchy 下界、半径/可行性、正实际下降、ρ&gt;0.1、三个尺度直接预测检查。最大预测相对误差 {maxerror:.5g}。"
    f"独立重算 {checks['candidate_count']} 个完整映射候选与 {checks['exhaustive_models_replayed']} 个小模型完成。本轮计算 {a['seconds']:.1f} s。</p>"
    '<p>当前 G7+A 的明确优势支持下一阶段采用完整历史库加持久化环带种子，在每个新状态重算响应，首先验证多步复用。'
    '本轮并未实施这段续算，不能直接确定重扫周期。'
    '旧种子收益衰减是可测试的触发假设，尚未验证；伴随实现的收益还需要重扫频率与成本证据。'
    '新颖度属于当前有限子空间，不能证明物理问题集中在 375–750 GHz，也不能推出全系统有效维数。'
    '尚未获得周期一根、稳定性认证或排除 multiple shooting 的证据。</p>')
payload=base64.b64encode(json.dumps(a,ensure_ascii=False,indent=2).encode()).decode()
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：完整历史库与环带新颖度验证</title><style>body{margin:0;background:#edf2f6;color:#182d40;font:17px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:25px}section{background:white;border-top:4px solid #176c91;padding:28px;margin:25px 0}h1{font-size:30px}h2{font-size:24px;margin-top:0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cddbe4;padding:8px;text-align:left}th{background:#e8f1f6}.table{overflow:auto}a{color:#176c91}@media(max-width:600px){main{padding:10px}section{padding:14px}h1{font-size:24px}h2{font-size:20px}}</style><main><h1>原生 10 MHz：完整历史库与环带新颖度验证</h1>'+''.join(parts)+f'<p><a download="history_complete.json" href="data:application/json;base64,{payload}">下载完整数值记录</a></p></main></html>'
(ROOT/'原生10MHz_完整历史库与环带新颖度验证.html').write_text(html,encoding='utf8')
print('Sections',len(parts))
