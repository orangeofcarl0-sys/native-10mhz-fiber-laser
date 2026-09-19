"""Offline evidence report; preserve the previous map and diagnosis reports."""
import base64, html, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
                     'axes.unicode_minus':False,'font.size':12})
pilot = json.loads((ROOT/'steady_pilot.json').read_text(encoding='utf8'))
follow = json.loads((ROOT/'steady_followup.json').read_text(encoding='utf8'))
validation = json.loads((ROOT/'steady_validation.json').read_text(encoding='utf8'))
rows = pilot['trials'] + follow['trials']
names = ['初猜：10 ps / 100 pJ','初猜：30 ps / 100 pJ','初猜：30 ps / 400 pJ',
         '第二组末态：增加Krylov预算','第600圈保存态']
status = {'line_search_stalled':'线搜索停滞','iteration_budget_reached':'达到迭代预算',
          'residual_converged':'缩放残差收敛，仍须后续验收'}

def embed(fig, name):
    path = ROOT/(name+'.png')
    fig.savefig(path,dpi=160,bbox_inches='tight'); plt.close(fig)
    return '<img alt="'+html.escape(name)+'" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'

fig,ax = plt.subplots(figsize=(11,5.6))
for name,row in zip(names,rows):
    ax.semilogy([h['calls'] for h in row['history']],
                [h['residual'] for h in row['history']],label=name,marker='.',markersize=5)
ax.axhline(1e-7,color='black',ls='--',label=r'求解残差门槛 $10^{-7}$')
ax.set(xlabel='残差函数累计调用次数（不是物理圈数）',ylabel='联合缩放残差的二范数',
       title='不同初始状态与线性求解预算的残差变化')
ax.grid(alpha=.2); ax.legend(fontsize=10)
convergence = embed(fig,'联合残差收敛曲线')

fig,ax = plt.subplots(figsize=(11,5))
for name,row in zip(names[3:],rows[3:]):
    history=[h for h in row['history'] if 'linear_relative_residual' in h]
    ax.semilogy([h['step'] for h in history],
                [h['linear_relative_residual'] for h in history],marker='o',label=name)
ax.axhline(.03,color='black',ls='--',label='线性求解目标 0.03')
ax.set(xlabel='Newton步骤',ylabel='‖J Δx + R‖ / ‖R‖',title='Newton方向的线性方程求解误差')
ax.grid(alpha=.2);ax.legend(fontsize=10)
linear = embed(fig,'线性方程残差')

waveforms = ''
paths = [ROOT/f'steady_trial_{k}.npz' for k in range(1,4)] + [
    ROOT/'best_pilot_larger_krylov.npz',ROOT/'saved_RT600.npz']
for name,path in zip(names,paths):
    data = np.load(path); a=data['a']; dt=float(data['dt'])
    t=(np.arange(a.shape[-1])-a.shape[-1]/2)*dt
    fig,ax=plt.subplots(figsize=(11,4.5))
    ax.plot(t,np.sum(abs(a)**2,axis=0))
    ax.set(xlabel='腔内参考截面的时间 / ps',ylabel='瞬时功率 / W',title=name+'：求解停止时的腔内波形')
    ax.grid(alpha=.2)
    waveforms += embed(fig,path.stem)

table=''
for name,row in zip(names,rows):
    table += '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [name,
        status[row['status']],f"{row['residual']:.3g}",f"{row['relative_field_residual']:.3g}",
        f"{row['population_gap']:.3g}",f"{row['output_energy_nJ']:.4f}",row['peaks'],
        f"{row['time_edge']:.2g} / {row['spectral_edge']:.2g}"])+'</tr>'
validation_text='；'.join(f"{v['topology']}：CPU/GPU相对差 {v['cpu_gpu_relative']:.2e}，方向导数相对差 {v['jv_relative_discrepancy']:.2e}" for v in validation)
document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：周期一联合自洽求解试验</title>
<style>body{margin:0;background:#edf2f7;color:#172e43;font:17px/1.8 "Microsoft YaHei",sans-serif}main{max-width:1150px;margin:auto;padding:36px 22px}section{background:white;padding:30px;margin:24px 0;border-radius:10px}h1{font-size:32px}h2{font-size:25px;color:#155785}img{width:100%;height:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ced8e2;padding:10px;text-align:left}.scroll{overflow:auto}.note{padding:16px;border-left:4px solid #d28d28;background:#fff5df}code{background:#eef4f8;padding:3px 7px}a{color:#155785}</style><main>
<h1>原生10 MHz：周期一联合自洽求解试验</h1><p>固定总腔长20.42 m · CNT → OC · 输出耦合80% · 净色散+0.2 ps² · 泵浦50 mW</p>
<section><h2>1．本次求解回答什么问题</h2><p>检验能否在不改变EDF寿命和光学器件的情况下，直接求得每圈重复的非零脉冲。它与从初态出发的真实逐圈演化互补：自洽解存在、局部动态稳定、实际初态可达是三个不同的问题。</p><p class="note">本页按实际停止状态报告结果。未取得合格固定点的案例不计算Floquet稳定性，也不作为稳定分支延拓锚点；求解失败不证明物理上无解。目标仍是输出0.1–0.5 nJ的稳定单脉冲。</p></section>
<section><h2>2．慢增益残差如何改写</h2><p><code>N⁺−N = (N_eq−N)[1−exp(−B T_R)]</code>。周期一状态可等价求解 <code>N−N_eq=0</code>，去掉很小的每圈恢复因子。N_eq仍由完整腔内传播、沿程泵浦和当前反转共同决定。</p><p>同时求解复光场、15个反转单元、整体相位和时间位移。模板内积用于平滑固定相位/时间规范；线搜索保持0&lt;N&lt;1。没有强制重置脉冲能量。CNT跨窗恢复因子小于10⁻¹⁴时才允许省去跨圈状态。</p><p>固定8192点，时间间隔0.125 ps，窗宽1024 ps；EDF步长0.1 m，其余光纤最大步长0.5 m。时间窗远短于约100 ns腔周期，结果只覆盖该局部窗口模型。初猜宽度为高斯强度的标准差，不是FWHM；能量为初始腔内能量，不是指定输出。</p></section>
<section><h2>3．实现等价性与方向导数检查</h2><p>27项CPU单元与回归测试通过。两种拓扑均检查了冻结反转传播的光场、输出和动态反转增量的等价性；已知方程根用于验证Newton迭代。</p><p>VALIDATION</p><p>上述检查验证实现和局部差分，不代替实际稳态的网格收敛和物理稳定性检查。</p></section>
<section><h2>4．多初猜与增加线性求解预算的结果</h2><p>前三组最多20步Newton，每步两轮LGMRES、每轮25维内迭代；后两组提高到80维。后续试验重新固定模板，因此跨试验的绝对缩放残差不宜直接比较，应结合相对光场残差和反转误差。</p><div class="scroll"><table><tr><th>初态</th><th>停止原因</th><th>联合残差</th><th>相对光场残差</th><th>最大反转差</th><th>输出 / nJ</th><th>局部峰数</th><th>时窗 / 频谱边界能量占比</th></tr>TABLE</table></div>CONVERGENCE<p>输出能量和峰数均为未必收敛的迭代状态诊断，不能当作可用锁模输出。参考边界筛选值为时窗外缘占比≤10⁻⁸、频谱外缘占比≤10⁻¹⁰；超出提示需要更大窗口或更细时间网格，不能忽略。</p></section>
<section><h2>5．各次求解停止时的完整腔内波形</h2><p>以下各图采用完整时间窗，独立大图显示，未裁掉卫星峰或边界背景。这是参考截面的腔内功率；表中的能量在输出耦合器处计算。</p>WAVEFORMS</section>
<section><h2>6．线性求解误差与证据边界</h2>LINEAR<p>线性相对残差接近1表示Newton方向几乎没有解好局部线性方程，并不意味着找到了物理不稳定性。后续重启同时改变了模板和缩放，不是只改变Krylov预算的严格单因素对照。</p><p>联合残差目标为10⁻⁷。合格锚点还必须具有非零局域脉冲、足够小的时频边界占比，并通过原始动态映射和细网格复核。在此之前，“更多Newton步骤”或“单个局部峰”都不足以证明稳定锁模。</p><p>若线性残差长期不能达到设定精度，应优先研究光场—反转块预条件；若达到线性精度仍停滞，再检查初猜、分支和非线性全局化。当前不能仅凭求解停滞归因为物理不稳定。</p><p>原始证据：<a download="steady_pilot.json" href="PILOT">前三组JSON</a> · <a download="steady_followup.json" href="FOLLOW">后续对照JSON</a>。原有地图和诊断报告保持不变。</p></section></main></html>'''
for key,value in [('VALIDATION',validation_text),('TABLE',table),('CONVERGENCE',convergence),('LINEAR',linear),('WAVEFORMS',waveforms),
                  ('PILOT','data:application/json;base64,'+base64.b64encode((ROOT/'steady_pilot.json').read_bytes()).decode()),
                  ('FOLLOW','data:application/json;base64,'+base64.b64encode((ROOT/'steady_followup.json').read_bytes()).decode())]:
    document=document.replace(key,value)
path=ROOT/'原生10MHz_周期一联合自洽求解试验.html'
path.write_text(document,encoding='utf8')
print(path)
