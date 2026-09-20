"""Offline Chinese report: current-state model selection and nonlinear evidence."""
import base64,io,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':12})
a=json.loads((ROOT/'active_history.json').read_text());check=json.loads((ROOT/'independent_checks.json').read_text())
groups=['H3','G2','G3','G4','G3+A'];parts=[]
def section(title,text,draw=None):
    pic=''
    if draw:
        fig,ax=plt.subplots(figsize=(11,5.5));draw(ax);fig.tight_layout()
        buf=io.BytesIO();fig.savefig(buf,format='png',dpi=150)
        fig.savefig(ROOT/f'figure_{len(parts)+1:02d}.png',dpi=150);plt.close(fig)
        pic='<img alt="'+title+'" src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'">'
    parts.append('<section><h2>'+title+'</h2>'+text+pic+'</section>')
def table(headers,rows):
    return '<div class="table"><table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
def find(group,radius):return next(v for v in a['trials'] if v['group']==group and v['radius']==radius)
def percent(v):return f'{100*v:.3f}%'

section('研究问题、物理条件与比较对象',
    '<p>本轮只检验：在同一个 S3 末态，按当前预测下降选择历史方向，是否优于固定取最近三个。'
    '不续跑，不改变激光器参数。起点 ‖R‖='+f"{a['initial_residual']:.12g}"+'，R 为归一化单圈闭合残差；merit Φ=‖R‖²/2。'
    '这里的求解步不是腔内往返次数，也不是物理时间。</p>'
    '<p>总腔长 20.42 m、10 MHz、CNT→OC、OC=80%、净色散 +0.2 ps²、泵浦 27.5 mW、CNT Psat=40 W；'
    '网格、反转、规范和尺度沿用上一轮。C 预条件器在补空间仍使用 −I，完整 Arnoldi 搜索没有被硬截断到 ±375 GHz。</p>'
    +table(['组别','额外历史方向（均包含当前 C 方向和同一 Arnoldi 基底）'],[
        ['H3','最近 3 个：来自外迭代 17、18、28'],['G2 / G3 / G4','从保存的 7 个方向中逐个贪心选 2 / 3 / 4 个'],
        ['G3+A','G3 加上当前重新计算的 375–750 GHz 环带梯度']]))

section('当前状态的对照结果与选择依据',
    '<p><b>已观察到：主动选择有用，但 2–3 个历史方向尚未达到收益饱和。</b>'
    '三个半径都依次选择来源外迭代 18、28、14、1；第三、第四个方向仍带来很大收益。'
    'G3 实际下降仅达到 G4 的约 32–34%，没有达到预设 95% 标准。5% 停止规则在前四个增加中均不触发。</p>'
    '<p><b>当前环带探针仍有明显增益。</b>G3+A 实际下降为 G3 的约 11.9–12.5 倍、G4 的约 4 倍。'
    '因此这个状态值得保留或试用新的环带方向；目前数据不支持把它判为无用，也不能推导出固定的周期性扫描频率。'
    '本轮尚未比较 G4+A，不能声称 G3+A 是所有组合中的最优方案。</p>'
    '<p>来源外迭代 18 的历史方向在当前单独沿正向是上坡方向，却在三个半径都首先被选中。'
    '联合模型允许负系数及相互抵消，这直接说明不能用单独 slope 的符号作删除依据。'
    '下一步若继续，应先验证新方向加入后的受控续算；本轮没有实施续算或宣称已进入快速求根区。</p>')

section('历史方向如何进入当前受限模型',
    '<p>历史 dᵢ 是过去重建得到的归一化 C 梯度方向；它不是旧 Newton 步，也不是旧 Jdᵢ。'
    '本轮在同一个当前状态 x 上重新计算 J(x)dᵢ。输入矩阵 B=[Z,dC,dᵢ,…]，输出矩阵 Y=[VH,JdC,Jdᵢ,…]。'
    '选择使 ‖R+Yc‖²/2 最小的系数 c，同时约束实际状态步 ‖Bc‖≤Δ。</p>'
    '<p>两个正交 QR 压缩分别保持输出残差范数和输入状态范数；不是对系数 c 施加半径。'
    '小型模型与未压缩完整模型的对照测试通过。保留原 Arnoldi 响应，最终候选另用完整映射和方向差分验证。</p>'
    '<p>每次从剩余方向中选择预测下降最大的一个，允许反向和组合使用，因此不按单独 slope 正负或方向余弦预删。'
    '完整算至 G4；“本次增量不足已有预测下降 5% 则不加入”的规则仅作为另一个策略诊断，未截断对照。</p>')

def spectra(ax):
    for key,label in [('D','状态方向 D'),('JD','当前响应 JD')]:
        ax.semilogy(range(1,8),a['rank'][key]['relative'],'o-',label=label)
    ax.set(xlabel='奇异值序号',ylabel='相对首奇异值',xticks=range(1,8));ax.legend();ax.grid(alpha=.2)
section('七个历史方向及当前响应的奇异谱',
    '<p>分别对 D=[d₁,…,d₇] 和 JD 做 SVD，并分别除以各自首奇异值。相对谱衰减表示各矩阵的数值条件，'
    '并非激光器物理自由度计数。严格线性映射不能提高精确秩；相对奇异值和给定容差下的数值秩可以不同。</p>'+table(
        ['相对容差','D 数值秩','JD 数值秩'],[[t,a['rank']['D']['numerical_ranks'][t],a['rank']['JD']['numerical_ranks'][t]] for t in a['rank']['D']['numerical_ranks']]),spectra)

def gram(ax):
    im=ax.imshow(a['history_gram'],vmin=-1,vmax=1,cmap='RdBu_r');labels=a['history_outer_steps']
    ax.set_xticks(range(7),labels);ax.set_yticks(range(7),labels)
    ax.set(xlabel='生成方向的外迭代编号',ylabel='生成方向的外迭代编号')
    for i in range(7):
        for j in range(7):ax.text(j,i,f"{a['history_gram'][i][j]:.2f}",ha='center',va='center',fontsize=9)
    plt.colorbar(im,ax=ax,label='归一化方向内积')
section('历史方向之间的相似程度',
    '<p>接近 1 表示方向相似，不代表可以直接删除。较小的差方向可能经过当前 Jacobian 放大，也可能帮助与其他基向量抵消。'
    '本轮依据联合受限模型增量选择；本图只用于解释候选之间的冗余。</p>'
    +table(['方向来源外迭代','当前 slope = R·Jd','响应尺度检查最大相对差异'],
        [[a['history_outer_steps'][i],f"{a['directions'][str(i)]['slope']:.6g}",
          f"{max(a['directions'][str(i)]['relative_checks']):.4g}"] for i in range(7)]),gram)

def ladder(ax):
    for audit in a['greedy']:
        ax.plot(range(1,5),[r['prediction'] for r in audit['ladder']],'o-',label=f"Δ={audit['radius']}")
    ax.set(xlabel='已选历史方向数',ylabel='预测 merit 下降',xticks=range(1,5));ax.legend();ax.grid(alpha=.2)
selection=[]
for audit in a['greedy']:
    for level,row in enumerate(audit['ladder'],1):
        selection.append([audit['radius'],level,', '.join(str(a['history_outer_steps'][i]) for i in row['selected']),f"{row['prediction']:.7g}",percent(row['marginal_gain'])])
section('逐次贪心选择与每个新增方向的收益',
    '<p>每一级都枚举剩余候选；先按预测 merit 下降排序，尚未使用真实下降来挑选方向。'
    '表中编号对应上一轮生成方向的外迭代。边际增益=(新预测下降−旧预测下降)/旧预测下降。</p>'
    +table(['半径','历史数','已选来源编号','预测下降','本次边际增益'],selection),ladder)

section('5% 停止规则对应的历史数量',
    '<p>若将 5% 规则直接用于选择，应在第一次不足 5% 的增加之前停止。这个规则可能提前阻断后续组合收益，'
    '所以本轮仍强制算出后续 G2/G3/G4，用来检验停止规则自身是否合适。0 表示该规则连第一个历史方向也不保留。</p>'
    +table(['半径','按 5% 规则保留的方向数','完整对照已算至'],[[v['radius'],v['policy_count'],4] for v in a['greedy']]))

def bars(ax):
    xx=np.arange(3);w=.15
    for k,g in enumerate(groups):ax.bar(xx+(k-2)*w,[find(g,d)['actual'] for d in a['radii']],w,label=g)
    ax.set_xticks(xx,[str(d) for d in a['radii']]);ax.set(xlabel='信赖域半径',ylabel='完整映射实际 merit 下降',ylim=(0,1.2*max(r['actual'] for r in a['trials'])));ax.legend(ncol=5);ax.grid(axis='y',alpha=.2)
section('五组候选的完整腔映射实际下降',
    '<p>实际下降为 [‖R(x)‖²−‖R(x+s)‖²]/2，所有候选从同一个 x 出发。'
    '这是真实单步对照，不是连续迭代曲线，也不能直接换算为长期加速倍数。</p>'+table(
    ['半径','组别','预测下降','实际下降','ρ','通过'],[[r['radius'],r['group'],f"{r['prediction']:.7g}",f"{r['actual']:.7g}",f"{r['rho']:.5f}",'是' if r['passed'] else '否'] for r in a['trials']]),bars)

ratios=[]
for d in a['radii']:
    g4=find('G4',d);g3=find('G3',d);oracle=find('G3+A',d)
    for g in ['G2','G3']:
        row=find(g,d);ratios.append([d,g+' / G4',percent(row['prediction']/g4['prediction']),percent(row['actual']/g4['actual'])])
    ratios.append([d,'G3+A / G3',percent(oracle['prediction']/g3['prediction']),percent(oracle['actual']/g3['actual'])])
section('历史数量与环带探针的增益对照',
    '<p>G2/G3 达到 G4 的 95%，只支持当前状态、当前 7 个历史方向、当前三个半径内收益接近，'
    '不能证明有效曲率只有 2–3 维。G3+A 超过 G3 的 125% 且通过候选验证，说明这个状态加入环带方向有价值；'
    '尚不能证明每隔固定步数扫描环带是最优策略。环带 GHz 是光场包络频率偏移，不是测距射频频点。</p>'
    +table(['半径','比较','预测下降比','实际下降比'],ratios))

max_error=max(c['discrepancy'] for r in a['trials'] for c in r['checks'])
section('数值验证、证据范围与当前未解决问题',
    f"<p>{sum(r['passed'] for r in a['trials'])}/15 个原始候选通过：Cauchy 预测下界、半径、可行性、正实际下降、ρ&gt;0.1，"
    f"以及三个差分尺度的模型检查。最大预测相对差异为 {max_error:.6g}；15 个候选独立重算完成：{check['all_replayed']}。"
    f"本轮计算用时 {a['seconds']:.1f} s。</p>"
    '<p>历史方向改善说明联合搜索模型有可利用的信息，但尚未证明完整 quasi-Newton 曲率机制。'
    '大部分残差来自光场，不代表反转变量对光场的耦合可以忽略。此前 S2/S3 步长余弦接近，也不足以证明二者机制相同。'
    '当前起点仍在 10⁻³ 量级，未得到周期一根或稳定单脉冲认证。</p>'
    '<p>本轮没有继续 30 步、放开物理参数、实现 L-BFGS/伴随或 multiple shooting。'
    '是否采用主动历史、保留多少方向、何时使用环带，应以本页同半径真实下降及后续受控验证决定。</p>')

payload=base64.b64encode(json.dumps(a,ensure_ascii=False,indent=2).encode()).decode()
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：主动历史子空间单状态验证</title><style>body{margin:0;background:#edf2f6;color:#182d40;font:17px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:25px}section{background:white;border-top:4px solid #176c91;padding:28px;margin:25px 0;box-shadow:0 3px 14px #1231}h1{font-size:30px}h2{font-size:24px;margin-top:0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cddbe4;padding:8px;text-align:left}th{background:#e8f1f6}.table{overflow:auto}a{color:#176c91}@media(max-width:600px){main{padding:10px}section{padding:14px}h1{font-size:24px}h2{font-size:20px}}</style><main><h1>原生 10 MHz：主动历史子空间单状态验证</h1><p>固定 S3 末态 · 七个历史方向 · 五组方法 · 三个半径</p>'+''.join(parts)+f'<p><a download="active_history.json" href="data:application/json;base64,{payload}">下载本轮完整数值记录</a></p></main></html>'
(ROOT/'原生10MHz_主动历史子空间单状态验证.html').write_text(html,encoding='utf8')
print('Report sections:',len(parts))
