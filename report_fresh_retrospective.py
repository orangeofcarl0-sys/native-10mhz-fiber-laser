"""Chinese offline report: old/fresh seed controls at four saved states."""
import base64,io,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':12})
a=json.loads((ROOT/'fresh_retrospective.json').read_text());checks=json.loads((ROOT/'independent_checks.json').read_text());parts=[]
def table(headers,data):return '<div class="table"><table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in data)+'</table></div>'
def section(title,text,draw=None):
    img=''
    if draw:
        fig,ax=plt.subplots(figsize=(11,5.5));draw(ax);fig.tight_layout();b=io.BytesIO();fig.savefig(b,format='png',dpi=150);fig.savefig(ROOT/f'figure_{len(parts)+1:02d}.png',dpi=150);plt.close(fig)
        img='<img alt="'+title+'" src="data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()+'">'
    parts.append('<section><h2>'+title+'</h2>'+text+img+'</section>')
labels=[str(s['step']) for s in a['states']];xx=np.arange(4)
section('四个保存状态与对照条件',
    '<p>问题只有一个：旧环带种子失去增益后，在同一状态重新计算的环带梯度是否能恢复大增益？'
    '使用保存轨迹的零起算 step 5、11、20、27，即上一报告第 6、12、21、28 步出发状态；四者均为 fresh-preconditioner 状态。'
    '不是取上一报告第 5、11、20、27 步之后的混合编号。</p>'
    '<p>每个状态固定历史库、当前 C、Arnoldi 基底和接受半径，只替换环带方向。'
    'C 与 Arnoldi 从保存状态确定性重建，先复现旧 seed 的预测与实际步，再计算 fresh 梯度。'
    '参数仍为总腔长 20.42 m、10 MHz、CNT→OC、OC=80%、净色散 +0.2 ps²、泵浦 27.5 mW、Psat=40 W。'
    '没有继续轨迹、调整触发条件、更新历史库或优化 fresh 半径。</p>'
    +table(['保存 step','报告行号','起点 ‖R‖','固定半径','历史方向数'],[[s['step'],s['report_step'],f"{s['residual']:.9g}",s['radius'],len(s['history_ids'])] for s in a['states']]))
section('预设判据与本次决策结果',
    f"<p>预设条件 Gold&lt;1.1 且 Gfresh&gt;1.5，共 {a['threshold_count']}/4 个状态满足；通过完整候选核验的恢复状态为 {a['validated_count']}/4。"
    +('<b>达到至少 3/4 的 adjoint 启动门槛。</b>' if a['adjoint_decision_gate'] else '<b>未达到至少 3/4 的 adjoint 启动门槛。</b>')+'</p>'
    '<p>Gold=Δm(H+Aold)/Δm(H)，Gfresh=Δm(H+Afresh)/Δm(H)，分母为同一个无环带模型的预测下降。H 包含固定的 Arnoldi 基底、当前 C 方向和历史库。'
    '模型阈值与真实候选检查分开：新方向还必须满足半径、可行性、Cauchy 下界、三尺度模型一致和真实下降 ρ&gt;0.1。'
    '方向余弦和一阶斜率用于解释是否发生旋转；仅凭预测增益不直接推出机制。</p>')
def model_gain(ax):
    ax.bar(xx-.18,[s['Gold'] for s in a['states']],.36,label='旧 seed')
    ax.bar(xx+.18,[s['Gfresh'] for s in a['states']],.36,label='fresh seed')
    ax.axhline(1.5,color='gray',ls='--',label='fresh 门槛 1.5');ax.set_xticks(xx,labels)
    ax.set(xlabel='保存轨迹 step（零起算）',ylabel='相对 history-only 的预测下降倍数');ax.legend();ax.grid(axis='y',alpha=.2)
section('旧方向与新方向的模型增益',
    '<p>两组均使用同一个 H。新方向由当前 375–750 GHz 环带的完整输出中心差分梯度归一化得到；旧方向保持原字节不变。'
    '这里的 GHz 是光场包络频率偏移，不是测距射频通道。</p>'
    +table(['step','Gold','Gfresh','满足阈值','真实核验恢复'],[[s['step'],f"{s['Gold']:.7g}",f"{s['Gfresh']:.7g}",s['threshold_met'],s['validated_recovery']] for s in a['states']]),model_gain)
def actual(ax):
    for i,(g,label) in enumerate([('history','仅历史'),('old','旧 seed'),('fresh','fresh seed')]):
        ax.bar(xx+(i-1)*.25,[next(t['actual'] for t in s['trials'] if t['group']==g) for s in a['states']],.25,label=label)
    ax.set_xticks(xx,labels);ax.set(xlabel='保存轨迹 step',ylabel='完整腔映射实际 merit 下降');ax.legend();ax.grid(axis='y',alpha=.2)
section('完整映射的真实单步下降',
    '<p>真实下降为 [‖R(x)‖²−‖R(x+s)‖²]/2。每个候选只做一次反事实映射，不把候选写回原轨迹；'
    '不同 step 是四个独立起点，不连成一条新的续算轨迹。</p>'
    +table(['step','组别','预测下降','实际下降','ρ','通过'],[[s['step'],t['group'],f"{t['prediction']:.7g}",f"{t['actual']:.7g}",f"{t['rho']:.5f}",t['passed']] for s in a['states'] for t in s['trials']]),actual)
def cosines(ax):
    ax.plot(xx,[s['cos_x'] for s in a['states']],'o-',label='状态方向 cos_x')
    ax.plot(xx,[s['cos_J'] for s in a['states']],'s-',label='当前响应 cos_J');ax.axhline(0,color='gray')
    ax.set_xticks(xx,labels);ax.set(xlabel='保存轨迹 step',ylabel='归一化内积',ylim=(-1.05,1.05));ax.legend();ax.grid(alpha=.2)
section('新旧方向及响应的夹角',
    '<p>cos_x=dold·dfresh；cos_J=(Jdold)·(Jdfresh)/(‖Jdold‖‖Jdfresh‖)。接近 1 表示同向，接近 0 表示近正交，接近 −1 表示反向。'
    '两项都在同一个当前状态定义；状态方向旋转与响应旋转不必相同。四点 cos_x 均接近零，说明相对旧种子已失配；本次未测相邻 fresh 梯度之间的旋转速率。独立性本身仍不等于下降收益。</p>'
    +table(['step','cos_x','cos_J','fresh 一阶下降 −R·Jdfresh','旧方向 −R·Jdold'],[[s['step'],f"{s['cos_x']:.8g}",f"{s['cos_J']:.8g}",f"{s['fresh_descent']:.8g}",f"{s['old_descent']:.8g}"] for s in a['states']]),cosines)
def slopes(ax):
    ax.plot(xx,[s['fresh_descent'] for s in a['states']],'o-',label='fresh')
    ax.plot(xx,[s['old_descent'] for s in a['states']],'s-',label='old');ax.set_yscale('symlog',linthresh=1e-8)
    ax.set_xticks(xx,labels);ax.set(xlabel='保存轨迹 step',ylabel='一阶下降量 −R·Jd');ax.legend();ax.grid(alpha=.2)
section('fresh 梯度强度与旧方向衰减',
    '<p>新环带梯度使用完整残差输出进行内积，扫描 6144 个实坐标；当前 fresh 方向的一阶下降应接近该受限梯度范数。'
    '前向与中心差分也作对照。旧方向斜率接近零而新方向斜率恢复，才支持“旧方向已失配，新梯度仍有价值”的解释。</p>',slopes)
maxcheck=max(c['relative_difference'] for s in a['states'] for t in s['trials'] for c in t['checks'])
section('恢复与独立核验记录',
    f"<p>{checks['candidates_replayed']} 个候选的完整映射与压缩模型已独立重算。最大三尺度预测差异 {maxcheck:.5g}；"
    f"四点总计算时间 {a['seconds']:.1f} s。</p>"
    +table(['step','原旧步相对恢复误差','旧预测相对误差','梯度差分相对差异'],[[s['step'],f"{s['replay_step_error']:.4g}",f"{s['replay_prediction_error']:.4g}",f"{s['gradient_stencil_difference']:.4g}"] for s in a['states']])
    +'<p>保存了输入哈希、原始执行源码、完整候选步和压缩模型。核验覆盖旧对照恢复、实际下降、模型预测、新旧方向内积与 fresh 斜率。</p>')
section('下一阶段边界',
    ('<p>本次达到用户预设的启动门槛，下一阶段转入 Jᵀv/离散伴随开发准备：先逐算子建立实坐标 VJP，并通过点积恒等式与现有 streaming 梯度对照，再接入求解器。'
     '本轮没有实现伴随，也未测得其加速比；启动依据是重复出现的 fresh 梯度价值和已知 streaming 列数成本。</p>' if a['adjoint_decision_gate'] else
     '<p>本次未达到预设的全面启动门槛。应按各状态的实际恢复情况判断是部分恢复还是均无明显收益；不把近似阈值或局部结果扩展成全局结论。</p>')
    +'<p>没有调整重扫触发器或运行新轨迹。四个采样状态不足以确定通用重扫周期；不证明全频带梯度都重要，也不等于物理根存在、稳定单脉冲或 multiple shooting 已被排除。</p>')
payload=base64.b64encode(json.dumps(a,ensure_ascii=False,indent=2).encode()).decode()
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：四状态fresh环带回溯验证</title><style>body{margin:0;background:#edf2f6;color:#182d40;font:17px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:25px}section{background:white;border-top:4px solid #176c91;padding:28px;margin:25px 0}h1{font-size:30px}h2{font-size:24px;margin-top:0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cddbe4;padding:8px;text-align:left}th{background:#e8f1f6}.table{overflow:auto}a{color:#176c91}@media(max-width:600px){main{padding:10px}section{padding:14px}h1{font-size:24px}h2{font-size:20px}}</style><main><h1>原生 10 MHz：四状态 fresh 环带回溯验证</h1>'+''.join(parts)+f'<p><a download="fresh_retrospective.json" href="data:application/json;base64,{payload}">下载完整数值记录</a></p></main></html>'
(ROOT/'原生10MHz_四状态fresh环带回溯验证.html').write_text(html,encoding='utf8');print('Sections',len(parts))
