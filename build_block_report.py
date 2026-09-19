"""Standalone report of successful and unsuccessful preconditioning controls."""
import base64,json,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
                     'axes.unicode_minus':False,'font.size':12})
load=lambda name:json.loads((ROOT/name).read_text(encoding='utf8'))
sides=load('linear_sides.json')['trials']
coarse=load('linear_coarse.json')
selected=load('linear_selected.json')
wide=load('linear_wide.json')
newton=load('wide_newton.json')
paired=load('block_pilot.json')
variants=[('无预条件',sides[0],0),('平均光学块·左',sides[1],18),
          ('平均光学块·右',sides[2],18)]
variants += [(f"中心±{r['cutoff']}频点",r,r['construction_calls']) for r in coarse]
variants += [(f"选取{r['selected_bins']}频点",r,r['construction_calls']) for r in selected]
variants += [('连续±384频点',wide,wide['construction_calls'])]

def picture(fig,name):
    p=ROOT/(name+'.png');fig.savefig(p,dpi=160,bbox_inches='tight');plt.close(fig)
    return f'<img alt="{name}" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'

fig,ax=plt.subplots(figsize=(11,5.5))
labels=[v[0] for v in variants]
ax.barh(labels,[v[1]['relative_linear_residual'] for v in variants],color='#247da0')
ax.axvline(.03,color='#a94428',ls='--',label='线性求解门槛 0.03')
ax.set(xscale='log',xlabel='真实线性残差 ‖J Δx + R‖ / ‖R‖',title='同一状态、同一Krylov预算下的预条件对照')
ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend()
linear_fig=picture(fig,'预条件的真实线性残差对照')

support=np.load(ROOT/'spectral_support.npz');n=len(support['field_power'])
frequency=np.fft.fftshift(np.fft.fftfreq(n,.125))
fig,ax=plt.subplots(figsize=(11,5.5))
for key,label in [('field_power','光场频谱'),('residual_power','光场残差频谱')]:
    p=np.fft.fftshift(support[key]);ax.semilogy(frequency,np.maximum(p/p.max(),1e-16),label=label)
ax.axvspan(-64/1024,64/1024,color='#f3ba5a',alpha=.25,label='中心±64频点')
ax.axvline(384/1024,color='gray',ls='--');ax.axvline(-384/1024,color='gray',ls='--')
ax.set(xlim=(-.65,.65),ylim=(1e-10,2),xlabel='相对包络频率 / THz',ylabel='各自峰值归一化的谱功率',
       title='光场与未满足自洽条件的频谱分布')
ax.grid(alpha=.2);ax.legend();spectrum_fig=picture(fig,'光场与残差频谱')

fig,ax=plt.subplots(figsize=(11,5.5))
for label,row in [('无预条件成对基线',paired['trials'][0]),('连续频带右预条件',newton)]:
    ax.semilogy([v['step'] for v in row['history']],[v['residual'] for v in row['history']],marker='o',label=label)
ax.axhline(1e-7,color='black',ls='--',label=r'自洽残差门槛 $10^{-7}$')
ax.set(xlabel='Newton步骤（不是物理圈数）',ylabel='同一缩放定义的联合残差',title='完整非线性求解的残差变化')
ax.legend();ax.grid(alpha=.2);nonlinear_fig=picture(fig,'非线性残差变化')

fig,ax=plt.subplots(figsize=(11,4.8))
h=[v for v in newton['history'] if 'alpha' in v]
ax.semilogy([v['step'] for v in h],[v['alpha'] for v in h],marker='o',color='#a94428')
ax.set(xlabel='Newton步骤',ylabel='接受步长 α',title='线搜索对Newton更新幅度的限制')
ax.grid(alpha=.2);step_fig=picture(fig,'线搜索接受步长')

state=np.load(ROOT/'wide_newton.npz');t=(np.arange(n)-n/2)*.125
fig,ax=plt.subplots(figsize=(11,5))
ax.plot(t,np.sum(abs(state['output'])**2,axis=0),color='#247da0')
ax.set(xlabel='时间 / ps',ylabel='输出瞬时功率 / W',title='完整Newton求解停止时的输出波形')
ax.grid(alpha=.2);waveform_fig=picture(fig,'末态输出波形')

table=''
for name,row,construction in variants:
    # side comparison times exclude one shared optical-block construction.
    time_note='（不含构建）' if name.startswith('平均') else ''
    total=row['calls']+construction if name.startswith('平均') else row['calls']
    table+='<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in
        [name,f"{row['relative_linear_residual']:.4g}",construction,total,f"{row['seconds']:.2f}"+time_note])+'</tr>'
summary=(f"停止原因：达到12步预算，未收敛；联合残差 {newton['residual']:.4g}；相对光场残差 "
         f"{newton['relative_field_residual']:.4g}；最大反转自洽差 {newton['population_gap']:.4g}；"
         f"输出 {newton['output_energy_nJ']:.5f} nJ，局部峰数 {newton['peaks']}。")
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：光场—反转预条件与稳态求解</title>
<style>body{{margin:0;background:#edf2f7;color:#172e43;font:17px/1.8 "Microsoft YaHei",sans-serif}}main{{max-width:1150px;margin:auto;padding:36px 22px}}section{{background:white;padding:30px;margin:24px 0;border-radius:10px}}h1{{font-size:32px}}h2{{font-size:25px;color:#155785}}img{{width:100%;height:auto;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ced8e2;padding:10px;text-align:left}}.scroll{{overflow:auto}}.note{{padding:16px;border-left:4px solid #d28d28;background:#fff5df}}</style><main>
<h1>原生10 MHz：光场—反转预条件与稳态求解</h1>
<p>固定20.42 m腔长 · CNT → OC · OC 80% · 净色散+0.2 ps² · 泵浦50 mW</p>
<section><h2>1．本次算法对照回答什么问题</h2><p>检验预条件能否解决上一轮Krylov线性求解停滞，并进一步降低完整非线性残差。模型、初始状态、模板、缩放、网格及相位初估均固定；不缩短EDF寿命、不重置脉冲能量。</p><p class="note">本次改善了同一点的线性求解精度，但完整Newton结果尚未优于无预条件基线，也没有找到稳态单脉冲。当前50 mW案例用于验证求解器，不是已经确定的目标光源设计。</p></section>
<section><h2>2．不同预条件保留了哪些耦合</h2><p>第一版平均光学块保留线性色散、偏振、反转增益谱和损耗，结合反转/规范变量的Schur补，但没有精确表示局部Kerr和CNT微分。实测左预条件和右预条件均未优于无预条件，失败结果保留如下。</p><p>后续采用完整传播器在傅里叶子空间计算实雅可比：两种偏振的实部/虚部、15个反转、整体相位和时间位移一起求逆。所有差分探测仍使用原8192点细网格，仅预条件逆矩阵受子空间限制，完整残差与Krylov方向导数不截谱。</p><p>右预条件求解J M y = −R，之后Δx=M y；它保持原方程残差的意义。子空间外采用负单位近似。批量探测与逐点计算的相对差约7.5×10⁻¹⁵；每个批量案例都计入调用成本。</p></section>
<section><h2>3．为什么需要覆盖较远频谱分量</h2><p>光场的大部分能量位于中心，但当前自洽残差有大量能量在较远频率。只根据光场峰值或窄中心频带选预条件，容易漏掉尚未解好的分量。图中两条谱分别按各自峰值归一化，不能据此比较绝对能量。</p>{spectrum_fig}<p>中心±64频点仅覆盖约6%的残差谱能量；连续±384频点覆盖约99.93%。离散选择513个重要频点虽覆盖约98.77%，本次线性求解仍没有明显改善；尚不能把改善仅归因于某一个非线性项。</p></section>
<section><h2>4．同一点的线性求解结果与成本</h2><div class="scroll"><table><tr><th>方法</th><th>真实线性残差</th><th>构建探测数</th><th>总映射调用数</th><th>实测秒数</th></tr>{table}</table></div>{linear_fig}<p>连续±384频点对应3093维预条件，真实线性残差0.0186，通过0.03门槛。构建3093次映射，之后Krylov约5次映射，另有一次真实残差复核；总耗时约54秒。该结果支持线性精度改善，不支持“整体更快”的结论。不同原型有串行/批量构建差异，耗时不能作为只由频带选择造成的因果对照。</p></section>
<section><h2>5．完整Newton迭代与末态输出</h2><p>{summary}</p><p>宽带预条件每3步重建；若上一线性步骤真实残差大于0.1则提前重建。最多12步，基线最多10步且在第5步停滞；超过共同步骤范围不作为等预算性能对照。宽带运行耗时{newton['seconds']:.1f}秒、映射调用{newton['evaluations']}次。</p>{nonlinear_fig}<p>无预条件基线最终联合残差0.00838，宽带预条件为0.00971；当前完整求解没有显示性能收益。后期多数线性步骤达到约0.03的要求，但线搜索步长缩至1/32，说明不能继续仅用线性求解精度解释停滞。信赖域/正则化步、初猜及解分支结构仍需区分验证。</p>{step_fig}{waveform_fig}<p>末态时间边界能量占比{newton['time_edge']:.3g}，频谱边界占比{newton['spectral_edge']:.3g}；参考筛选上限分别为10⁻⁸和10⁻¹⁰。波形与能量是停止态诊断，不能自动视作可用锁模输出；1.92 nJ也不属于0.1–0.5 nJ目标区间。</p></section>
<section><h2>6．验证证据与结论范围</h2><p>34项测试通过，覆盖线性光学极限、稠密Schur补对照、傅里叶限制/提升的正交性、已知子空间线性系统、右预条件Newton已知解、批量与逐点残差一致性，以及原有物理模型回归。</p><p>未通过完整稳态验收时，不进行Floquet认证或稳定分支追踪；算法停滞不是物理无解证明。新预条件保留为可选实验入口，默认求解设置不切换。本报告独立保存，不覆盖前一版报告或删除原有页面。</p></section></main></html>'''
download='data:application/json;base64,'+base64.b64encode((ROOT/'wide_newton.json').read_bytes()).decode()
body=body.replace('</main>',f'<p><a download="wide_newton.json" href="{download}">下载完整Newton诊断JSON</a></p></main>')
(ROOT/'原生10MHz_光场反转预条件与稳态求解.html').write_text(body,encoding='utf8')
print(ROOT/'原生10MHz_光场反转预条件与稳态求解.html')
