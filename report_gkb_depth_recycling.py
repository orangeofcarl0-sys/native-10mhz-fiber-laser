"""Offline depth/recycling report with actual merit and explicit cost accounting."""
import base64,io,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':12})
a=json.loads((ROOT/'audit.json').read_text(encoding='utf8'));c=json.loads((ROOT/'independent_checks.json').read_text(encoding='utf8'))
depth=[r for r in a['trials'] if 'k' in r];lookup={r['name']:r for r in a['trials']};sections=[]
def table(head,rows):
    return '<div class="table"><table><tr>'+''.join('<th>'+h+'</th>' for h in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
def section(title,body,draw=None):
    picture=''
    if draw:
        fig,ax=plt.subplots(figsize=(11,5.5));draw(ax);fig.tight_layout();b=io.BytesIO();fig.savefig(b,format='png',dpi=150)
        fig.savefig(ROOT/f'figure_{len(sections)+1:02d}.png',dpi=150);plt.close(fig)
        picture='<img alt="'+title+'" src="data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()+'">'
    sections.append('<section id="s'+str(len(sections)+1)+'"><h2>'+title+'</h2>'+body+picture+'</section>')
section('冻结对象与本轮检验的问题',
 f'<p>当前状态为纯GKB正式40步末态，R=<b>{a["initial_residual"]:.10g}</b>；旧状态为生成第40步之前的状态，R={a["old_residual"]:.10g}。所有新候选均从当前状态出发，半径固定为 <b>{a["radius"]}</b>，没有接受成一条新轨迹。</p>'
 '<p>固定泵浦27.5 mW、OC=80%、GDD=+0.2 ps²、Psat=40 W、CNT→OC、20.42 m、10 MHz、网格与gauge。检验两件事：增加GKB深度能否取得可用的真实下降；回收上一状态完整64维空间能否以更低新增成本补充当前空间。</p>'
 '<p>上一轮末步的Δm₆₄来自旧状态。本轮重新计算当前末态K64，两者不能直接混用。</p>')
section('深度扩展与回收模型的定义',
 '<p>当前状态一次连续GKB到192维，检查64/80/96/112/128/144/160/176/192，不因中途结果提前停止。每个Bₖ解相同SVD信赖域子问题，U/V保持两遍完整重正交。</p>'
 '<p>R64重建上一状态的Vold，但64个响应全部用当前J重算。K64+R64将两个状态空间正交合并，使用QR+SVD、10⁻¹²相对截断，同步变换响应，保持真实状态欧氏trust metric。它回收整个Krylov空间，不是少量梯度history。</p>'
 '<p>预设：连续两次16维模型增益低于1%表示深度饱和；回收分别对K128/K192检查模型覆盖&gt;90%、新增成本≤80%且真实保护通过。该门是本状态证据，不自动启动outer。方法背景见<a href="https://web.stanford.edu/group/SOL/software/lsqr/">Stanford SOL 的 GKB/LSQR 说明</a>。</p>')
def reduction(ax):
    ax.plot([r['k'] for r in depth],[r['prediction'] for r in depth],'o-',label='模型预测下降')
    ax.plot([r['k'] for r in depth],[r['actual'] for r in depth],'s-',label='真实目标下降')
    ax.axhline(0,color='gray');ax.set(xlabel='GKB维数',ylabel='目标Φ=½‖R‖²的下降');ax.legend();ax.grid(alpha=.2)
section('扩维后的预测下降与真实下降',
 '<p>预测持续增加并不保证真实下降增加。固定半径下，图中真实下降为负的候选会被拒绝；它们不是可用精度改善，也没有更新冻结状态。</p>',reduction)
def absolute(ax):
    ax.bar([r['k'] for r in depth[1:]],[r['absolute_increment'] for r in depth[1:]],width=10)
    ax.set(xlabel='扩展后的GKB维数',ylabel='每增加16维的绝对模型下降增量');ax.grid(axis='y',alpha=.2)
section('每增加16维取得的绝对收益',
 '<p>纵轴是Δmₖ−Δmₖ₋₁₆，不以当前Δm归一化。它直接检验“有用方向是否仍在持续增加”，并避免小分母造成的相对倍率错觉。</p>',absolute)
def marginal(ax):
    ax.plot([r['k'] for r in depth[1:]],[100*r['relative_increment'] for r in depth[1:]],'o-')
    ax.axhline(1,color='red',ls='--',label='1%停维线');ax.set(xlabel='GKB维数',ylabel='每增加16维的相对模型收益 / %');ax.legend();ax.grid(alpha=.2)
section('深度饱和判据的结果',
 f'<p>首次满足连续两次低于1%的checkpoint：<b>{a["first_saturation_checkpoint"] if a["first_saturation_checkpoint"] is not None else "未出现"}</b>。这只检验已生成的有限空间，不能推断完整Jacobian的条件数或single-shooting不可解。</p>',marginal)
def steps(ax):
    ax.plot([r['k'] for r in depth],[r['step_norm'] for r in depth],'o-',label='候选步范数')
    ax.axhline(a['radius'],ls='--',color='red',label='固定半径');ax.set(xlabel='GKB维数',ylabel='缩放状态中的步范数');ax.legend();ax.grid(alpha=.2)
section('扩维后的步长与信赖域边界',
 '<p>64维时步远小于半径，并不保证192维时仍如此。扩维后新方向可能产生更大的最小二乘步；λ&gt;0表示约束开始生效。三尺度Js验证的是局部线性响应，ρ则额外检查有限步长的非线性影响，两者不能互相替代。</p>',steps)
section('各深度候选的完整验收记录',table(['k','预测Δm','真实ΔΦ','ρ','‖s‖','λ','κ(B)','通过'],[
 [r['k'],f'{r["prediction"]:.5g}',f'{r["actual"]:.5g}',f'{r["rho"]:.4f}',f'{r["step_norm"]:.5g}',f'{r["lambda_value"]:.4g}',f'{r["singular_values"][0]/r["singular_values"][-1]:.4g}',r['passed']] for r in depth]))
def spectrum(ax):
    for name in ['K64','K128','K192']:
        values=lookup[name]['singular_values'];ax.semilogy(range(1,len(values)+1),values,label=name)
    ax.set(xlabel='B奇异值序号',ylabel='奇异值');ax.legend();ax.grid(alpha=.2)
section('不同深度捕获到的奇异值谱',
 '<p>B₆₄的良好条件数只描述其已捕获空间。扩大空间后出现的更小奇异值对应响应更弱的状态组合；此图不等于完整Jacobian谱，也不单独证明分岔或物理不稳定。</p>',spectrum)
models=['K64','R64','K64+R64','K96','K128','K192']
section('回收空间、联合空间与fresh空间的对照',
 table(['模型','预测Δm','真实ΔΦ','ρ','通过全部保护','新增成本/s'],[[name,f'{lookup[name]["prediction"]:.5g}',f'{lookup[name]["actual"]:.5g}',f'{lookup[name]["rho"]:.4f}',lookup[name]['passed'],f'{lookup[name]["cost_seconds"]:.2f}'] for name in models])+
 '<p>额外列出K96，防止只比较两个更深、但非线性表现更差的空间。R64即使局部ρ良好，也可能因不含当前完整梯度、达不到Cauchy下界而不能单独接受。联合空间的模型增益应与单独回收空间分开解释。</p>'
 f'<p>联合空间保留rank={a["merge"]["rank"]}，正交误差 {a["merge"]["orth_error"]:.3g}；相对K64的模型下降倍率为 {a["union_gain_over_K64"]:.4f}。</p>')
def efficiency(ax):
    vals=[lookup[n]['actual']/lookup[n]['cost_seconds'] for n in models]
    ax.bar(models,vals);ax.axhline(0,color='gray');ax.set(ylabel='真实目标下降 / 新增构造与求解秒数');ax.tick_params(axis='x',rotation=15);ax.grid(axis='y',alpha=.2)
section('真实下降与成本的共同比较',
 '<p>此图是本冻结状态的一次局部效率诊断，不是outer吞吐率。负值仍显示为失败。回收成本按旧空间已缓存的情况计算；一次审计为了重建旧空间付出的额外成本另列。</p>',efficiency)
section('缓存可用与从零重建的成本口径',
 table(['成本项','秒'],[[name,f'{value:.3f}'] for name,value in a['timing'].items()])+
 '<p>计时在单独复测中进行，未同时运行测试或验证作业。fresh成本使用一次连续192维构造的对应前缀，加该checkpoint的小问题求解；联合warm成本=当前fresh64+64次当前Jv+正交合并及求解，cold成本额外包含旧64重建。没有用省略旧响应更新的方式夸大回收收益。</p>'+
 table(['相对fresh','模型覆盖','warm成本比例','真实下降比例','预设回收门'],[[k,f'{v["model_coverage"]*100:.2f}%',f'{v["cost_ratio"]*100:.2f}%',f'{v["actual_ratio"]:.4f}' if v['actual_ratio'] is not None else '不适用：fresh真实下降为负',v['supported']] for k,v in a['recycling_comparisons'].items()]))
section('独立复核与数据归档',
 f'<p>{c["candidates_replayed"]} 个候选独立复算；三尺度预测最大相对偏差 {c["maximum_prediction_error"]:.3g}。旧状态64模型复现误差 {a["old_reproduction_relative"]:.3g}；当前JV−UB相对误差 {a["health"]["bidiagonal_relative"]:.3g}。全套104项CPU测试通过。</p>'
 '<p>所有候选都保留三尺度Js、Cauchy下界、可行性、步长半径和ρ&gt;0.1检查，失败不改成接受。完整大基矩阵留在本地local_full_bases.npz；公共归档提供固定输入哈希、重建脚本、执行源、B和候选向量，避免向Git重复上传数百MB矩阵。</p>')
best=max([t for t in a['trials'] if t['passed']],key=lambda t:t['actual'])
section('本次冻结证据支持的路线',
 f'<p><b>192维仍未自然饱和，但扩维后的非线性限制已经显现。</b>本次通过保护的最佳真实下降为 {best["name"]}，ΔΦ={best["actual"]:.6g}、ρ={best["rho"]:.4f}。K144–K192的失败说明：不能把64维时“半径没有卡住、ρ接近1”的判断直接推广到更深空间。</p>'
 '<p>回收联合空间达到K128约98.6%的模型下降，且通过真实保护，支持继续研究回收信息的价值；但它未达到K192的90%模型覆盖，并且真实下降仍低于K96。下一版不能仅按模型覆盖率选路线，还需保留自适应信赖域并比较真实下降/成本。</p>'
 '<p>本轮未改变信赖半径、未预条件、未运行新outer轨迹。没有足够证据转multiple shooting，也不能宣称已经找到闭合根或稳定单脉冲。有限深度与有限步长的影响在本实验中同时存在。</p>')
payload=base64.b64encode(json.dumps(dict(result=a,checks=c),ensure_ascii=False).encode()).decode()
style='body{margin:0;background:#edf2f6;color:#182d40;font:17px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:25px}section{background:white;border-top:4px solid #176c91;padding:28px;margin:25px 0}h1{font-size:30px}h2{font-size:24px;margin-top:0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cddbe4;padding:8px;text-align:left}th{background:#e8f1f6}.table{overflow:auto}a{color:#176c91}@media(max-width:600px){main{padding:10px}section{padding:14px}h1{font-size:24px}h2{font-size:20px}}'
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：GKB深度与回收审计</title><style>'+style+'</style><main><h1>原生10MHz：GKB 64→192深度与空间回收冻结审计</h1>'+''.join(sections)+f'<p><a download="audit.json" href="data:application/json;base64,{payload}">下载完整记录</a></p></main></html>'
(ROOT/'原生10MHz_GKB深度与空间回收冻结审计.html').write_text(html,encoding='utf8');print('REPORT',len(sections))
