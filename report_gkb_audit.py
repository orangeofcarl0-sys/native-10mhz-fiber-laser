"""Self-contained Chinese report of the frozen GKB experiment."""
import base64,io,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':12})
a=json.loads((ROOT/'audit.json').read_text(encoding='utf8'))
c=json.loads((ROOT/'independent_checks.json').read_text(encoding='utf8'))
sections=[];names={'A':'A：root + 梯度 + H','B':'B：GKB','C':'C：GKB + H','D':'D：GKB + H + Z'}
def table(head,rows):
    return '<div class="table"><table><tr>'+''.join('<th>'+s+'</th>' for s in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(s)+'</td>' for s in row)+'</tr>' for row in rows)+'</table></div>'
def section(title,body,draw=None):
    picture=''
    if draw:
        fig,ax=plt.subplots(figsize=(11,5.5));draw(ax);fig.tight_layout();b=io.BytesIO()
        fig.savefig(b,format='png',dpi=160);fig.savefig(ROOT/f'figure_{len(sections)+1:02d}.png',dpi=160);plt.close(fig)
        picture='<img alt="'+title+'" src="data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()+'">'
    sections.append('<section id="s'+str(len(sections)+1)+'"><h2>'+title+'</h2>'+body+picture+'</section>')
def baseline(radius):return next(t for t in a['trials'] if t['arm']=='A' and t['radius']==radius)
def final(name):return [t for t in a['trials'] if t['arm']==name and (name=='A' or t['k']==32)]
section('本轮问题、固定末态与决策门',
 f'<p>起点为伴随梯度 40 步续算末态，残差范数 <b>{a["initial_residual"]:.10g}</b>。本轮只改变搜索子空间，所有候选都从同一个状态出发，不形成新轨迹。</p>'
 '<p>泵浦 27.5 mW、OC=80%、净色散 +0.2 ps²、Psat=40 W、CNT→OC、总腔长 20.42 m、重频 10 MHz、网格及规范模板保持原样。信赖半径作用于既有缩放状态的欧氏范数，不是物理能量。</p>'
 f'<p>预设门：B32 或 C32 在至少两个半径通过全部保护，且真实下降超过 A 的 1.5 倍。结果：B 为 {a["decision_counts"]["B"]}/3，C 为 {a["decision_counts"]["C"]}/3，<b>{"满足" if a["upgrade_gate"] else "未满足"}</b>升级门。本报告保留未通过候选，不按真实下降事后挑选维数。</p>')
section('最小二乘子空间的数学含义',
 '<p>目标是 Φ(x)=½‖R(x)‖²；局部模型是 ½‖R+Js‖²。GMRES 本身也最小化它所生成空间中的线性残差。本轮差别不是把“错误目标”换成“正确目标”，而是把右预条件 root Arnoldi 的主体空间换成由 J 与 Jᵀ 共同生成的 GKB 空间。</p>'
 '<p>b=−R，u₁=b/‖b‖；交替计算 Jᵀu 与 Jv，并对 U、V 各做两遍完整重正交。得到 JV≈UB，s=Vy，约束直接是 ‖y‖≤Δ。第一维 v₁=−JᵀR/‖JᵀR‖，因此 k=1 必须复现 full-gradient Cauchy。</p>'
 '<p>小矩阵 B=PΣQᵀ，y=Q diag[σ/(σ²+λ)]Pᵀ(βe₁)。用 SVD 与标量乘子求根满足半径，不形成完整 JᵀJ，也不对正规方程做 LU。GKB 不显式平方条件数，但并未消除原问题的奇异值病态。</p>'
 '<p>方法依据：<a href="https://web.stanford.edu/group/SOL/software/lsqr/">Stanford SOL：Paige–Saunders LSQR / Golub–Kahan</a>。实现为 steady_gkb.py；完整腔残差及伴随保持不变。</p>')
section('四组对照与历史库的统一方式',
 table(['组','搜索空间'],[[names[n],s] for n,s in [('A','Z + 当前完整负梯度 + 12方向H'),('B','Vₖ'),('C','Vₖ + 同一H'),('D','Vₖ + 同一H + 同一Z')]])+
 f'<p>A 在本末态新建 C 预条件器，沿用 Arnoldi 上限 240、线性相对容限 0.008，得到 Z 维数 {a["root_dimension"]}。这不声称重现上一轨迹的未保存缓存分解。C 仍只作预条件器，不计算 C-gradient。</p>'
 '<p>末态待处理库有 13 个方向。仅在 Δ=0.00625 用 A 的原 leave-one-out 规则裁成 12 个，然后所有组、所有半径共用它。GKB 不预条件；C、D 合并非正交方向后仍沿用实际状态度量的 SVD 白化。</p>'
 '<p>历史方向：'+', '.join(a['history_ids'])+'。</p>')
def actualplot(ax):
    for i,n in enumerate(names):
        ax.bar(np.arange(2)+(i-1.5)*.2,[t['actual']/baseline(t['radius'])['actual'] for t in final(n)[:2]],width=.19,label=names[n])
    ax.axhline(1.5,ls='--',color='black',label='预设 1.5 倍门');ax.set_xticks(range(2),[str(v) for v in a['radii'][:2]]);ax.set(xlabel='信赖半径 Δ',ylabel='真实目标下降 / A 的真实目标下降');ax.legend(fontsize=10);ax.grid(axis='y',alpha=.2)
section('32维候选的真实目标下降',
 '<p>纵轴比较 ΔΦ=½(‖R(x)‖²−‖R(x+s)‖²)，不是残差范数下降的倍率。A 是同半径分母，B/C/D 均固定使用 k=32。最大半径下 A 的真实下降为负、验收失败，因此不画该半径倍率；原始值完整保留于下表。</p>',actualplot)
section('32维候选的验收记录',table(['组','Δ','预测下降','真实下降','ρ','相对A','通过'],[
 [n,t['radius'],f'{t["prediction"]:.6g}',f'{t["actual"]:.6g}',f'{t["rho"]:.5f}',f'{t["actual"]/baseline(t["radius"])["actual"]:.4f}' if baseline(t['radius'])['actual']>0 else '不适用：A失败',t['passed']] for n in names for t in final(n)]))
def ladder(ax):
    for radius in a['radii']:
        rows=[t for t in a['ladder'] if t['radius']==radius]
        ax.plot([t['k'] for t in rows],[t['prediction']/baseline(radius)['prediction'] for t in rows],'o-',ms=3,label=f'Δ={radius}')
    ax.axhline(1,ls='--',color='gray');ax.set(xlabel='GKB 维数 k',ylabel='GKB 模型下降 / A 模型下降');ax.legend();ax.grid(alpha=.2)
section('GKB扩维带来的模型下降',
 '<p>计算全部 1–32 维，显示同半径归一化的预测。k=1 提供完整 Cauchy，之后的增益才是新增最小二乘曲率方向的贡献。没有使用 root 方程残差门限提前终止 GKB。</p>',ladder)
def extras(ax):
    for n in ['B','C','D']:
        rows=[t for t in a['trials'] if t['arm']==n and t['radius']==.00625]
        ax.plot([t['k'] for t in rows],[t['actual']/baseline(.00625)['actual'] for t in rows],'o-',label=names[n])
    ax.axhline(1.5,ls='--',color='gray');ax.set(xlabel='GKB 维数 k',ylabel='真实下降 / A（Δ=0.00625）');ax.legend();ax.grid(alpha=.2)
section('历史与原root空间的额外贡献',
 '<p>在中间半径查看 B→C→D：加入历史是否仍有效，以及加入原 Z 是否还能提高真实下降。嵌套空间的预测应不下降，但真实非线性下降不保证单调，因此必须同时看 ρ 与独立 Js。</p>',extras)
def marginal(ax):
    for row in c['saturation']:
        ax.plot([t['k'] for t in row['marginal']],[100*t['gain'] for t in row['marginal']],'o-',label=f'Δ={row["radius"]}')
    ax.axhline(1,ls='--',color='gray');ax.set(xlabel='GKB 维数 k',ylabel='每增加4维的相对模型增益 / %');ax.legend();ax.grid(alpha=.2)
section('扩维饱和的诊断',
 '<p>指标为 [Δmₖ−Δmₖ₋₄]/Δmₖ。把“连续若干次”明确为连续三次低于 1%，仅作诊断，仍完整运行到 32 维。</p>'+table(['Δ','首次连续三次满足的k'],[[t['radius'],t['first_three_consecutive']] for t in c['saturation']]),marginal)
section('算子一致性和真实性保护',
 table(['核对项','观测'],[[k,f'{v:.4g}'] for k,v in a['gkb_health'].items()])+
 f'<p>独立复算 {c["candidates_replayed"]} 个保存候选；起点与最新末态逐元素相同。GKB 小问题 KKT 最大相对误差 {c["maximum_gkb_kkt_relative"]:.3g}；名义 B 预测与保存 JV 直接预测的最大相对差 {c["maximum_bidiagonal_dense_prediction_relative"]:.3g}。</p>'
 '<p>每个候选必须满足 Cauchy 预测下界（1−10⁻⁶）、步长半径、可行性、真实下降为正且 ρ&gt;0.1。按 h=10⁻⁵、3×10⁻⁶、10⁻⁶ 做归一化中心差分 Js，要求各预测为正且与模型差异小于 5%。没有因为改用伴随而放宽门限。</p>'
 '<p>这是固定半径的原始候选审计。失败候选如 A 的最大半径步被明确拒绝，没有以 fallback 冒充该子空间的成功；正式 outer 中原有缩半径和 Cauchy fallback 仍应保留。</p>')
section('实际计算成本与比较边界',
 table(['项目','同步墙钟/s'],[[n,f'{a["timing"][n]["median"]:.4f}'] for n in ['forward','Jv','JTv']]+[[n,f'{a["timing"][n]:.3f}'] for n in ['gkb_build','c_build','root_arnoldi','history_responses']])+
 '<p>前三项是预热后三次中位数；完整 GKB 与 C/Arnoldi 为本次单次同步墙钟，包含各自 CPU 开销。Jᵀv 当前包含原映射重放。完整 GKB 使用 32 次 Jv 与 32 次 Jᵀv。此表比较空间构造，不把校验、全部75候选的成本算作一次 outer 的成本，也不等同于长程加速比。</p>'
 f'<p>整个审计耗时 {a["seconds"]:.1f} s。原 Z 对 full-gradient 的覆盖：'+', '.join(f'{t["span"]}={t["coverage"]:.5g}' for t in a['gradient_coverage'])+'；GKB 首列已包含该方向。</p>')
section('结论适用范围与下一阶段',
 f'<p>预设进入 outer 门：<b>{"通过" if a["upgrade_gate"] else "未通过"}</b>。结论仅对应本末态、当前缩放和三个半径。即使收益小，也不能单凭这一个状态宣告搜索空间已不是瓶颈；还要结合扩维是否饱和、数值一致性、历史和 Z 的边际价值。</p>'
 '<p>观测：B32 在前两个半径的真实下降分别约为 A 的 35.74 倍、55.04 倍，ρ≈1.005–1.008；C32 相对 B32 的真实增益仅约 0.009%–0.036%。D32 在中间半径比 B32 多约 3.73%，但需保留昂贵的 C 构造。本状态的证据支持优先以纯 GKB 作为下一轮主算法，历史和 Z 的小幅增益不足以单独证明值得永久保留。</p>'
 '<p>B32 的中间与最大半径结果完全相同，是因为无约束步范数约 0.006040，均位于半径内，并非漏算。该单步残差约 0.00107834，仍未到 0.001，更没有达到 10⁻⁷。</p>'
 '<p>本轮没有参数释放、annulus、period-2、Floquet、multiple shooting，也没有再跑 40 步。所有结果都是一次线性子空间审计，不是稳态求根成功或稳定单脉冲认证。</p>')
payload=base64.b64encode(json.dumps(dict(result=a,checks=c),ensure_ascii=False).encode()).decode()
style='body{margin:0;background:#edf2f6;color:#182d40;font:17px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:25px}section{background:white;border-top:4px solid #176c91;padding:28px;margin:25px 0}h1{font-size:30px}h2{font-size:24px;margin-top:0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cddbe4;padding:8px;text-align:left}th{background:#e8f1f6}.table{overflow:auto}a{color:#176c91}@media(max-width:600px){main{padding:10px}section{padding:14px}h1{font-size:24px}h2{font-size:20px}}'
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：GKB冻结末态审计</title><style>'+style+'</style><main><h1>原生10MHz：Gauss–Newton Krylov 冻结末态审计</h1>'+''.join(sections)+f'<p><a download="audit.json" href="data:application/json;base64,{payload}">下载完整75候选记录</a></p></main></html>'
(ROOT/'原生10MHz_GKB冻结末态子空间审计.html').write_text(html,encoding='utf8')
print('REPORT',len(sections))
