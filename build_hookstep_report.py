"""Independent diagnostics and controlled globalization results in offline HTML."""
import base64,json,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
                     'axes.unicode_minus':False,'font.size':12})
read=lambda name:json.loads((ROOT/name).read_text(encoding='utf8'))
taylor=read('taylor_diagnostics.json');window=read('window_diagnostics.json')
coarse=read('coarse_stationarity.json');pair=read('hookstep_pair.json');replay=read('original_replay.json')
labels={'line_search':'共享Arnoldi＋线搜索','hookstep':'共享Arnoldi＋Hookstep',
        'hookstep_reanchor':'Hookstep＋规范重设'}
status={'iteration_budget_reached':'达到迭代预算，未收敛','residual_converged':'缩放残差收敛，仍需物理/网格验收',
        'globalization_stalled':'全局化尝试停滞','radius_floor_reached':'信赖半径达到下限'}

def picture(fig,name):
    path=ROOT/(name+'.png');fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
    return f'<img alt="{name}" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'

fig,ax=plt.subplots(figsize=(11,5.2))
for row in taylor:
    ax.loglog([v['epsilon'] for v in row['epsilon_sweep']],
              [v['forward_relative'] for v in row['epsilon_sweep']],marker='o',label=f"原第{row['step']}步")
ax.set(xlabel='单位方向上的有限差分位移 ε',ylabel='前向Jd与独立中心差分的相对差',title='差分导数对扰动尺度的敏感性')
ax.grid(alpha=.2);ax.legend();epsilon_fig=picture(fig,'差分扰动尺度检查')

fig,ax=plt.subplots(figsize=(11,5.2))
for row in taylor:
    points=[v for v in row['taylor'] if 'forward_taylor' in v]
    ax.loglog([v['h'] for v in points],[v['forward_taylor'] for v in points],marker='o',label=f"原第{row['step']}步")
ax.set(xlabel='沿完整Newton方向的步长系数 h',ylabel='前向Taylor余项 / (h ‖Jd‖)',title='局部线性模型的适用步长范围')
ax.grid(alpha=.2);ax.legend();taylor_fig=picture(fig,'前向Taylor余项')

fig,ax=plt.subplots(figsize=(11,5.2))
for row in taylor:
    points=[v for v in row['taylor'] if 'symmetric_taylor' in v]
    ax.loglog([v['h'] for v in points],[v['symmetric_taylor'] for v in points],marker='o',label=f"原第{row['step']}步")
ax.set(xlabel='对称扰动系数 h',ylabel='‖R(x+hd)−R(x−hd)−2hJd‖ / (2h ‖Jd‖)',title='对称Taylor余项的步长依赖')
ax.grid(alpha=.2);ax.legend();symmetric_fig=picture(fig,'对称Taylor余项')

fig,ax=plt.subplots(figsize=(11,5.2))
for row in pair['trials']:
    ax.semilogy([h['step'] for h in row['history']],[h['residual'] for h in row['history']],marker='o',label=labels[row['name']])
ax.axhline(1e-7,color='black',ls='--',label=r'自洽门槛 $10^{-7}$')
ax.set(xlabel='外层步骤（不是物理圈数）',ylabel='联合缩放残差',title='同一初值的全局化方法对照')
ax.grid(alpha=.2);ax.legend();convergence_fig=picture(fig,'全局化方法残差对照')

fig,ax=plt.subplots(figsize=(11,5.2))
for row in pair['trials']:
    if row['name']=='line_search':continue
    accepted=[h for h in row['history'] if 'accepted_trial' in h]
    ax.semilogy([h['step'] for h in accepted],
                [h['trials'][h['accepted_trial']]['radius'] for h in accepted],marker='o',label=labels[row['name']])
ax.set(xlabel='接受更新的外层步骤',ylabel='接受步所用的信赖半径 Δ',title='实际缩放状态坐标中的信赖半径')
ax.grid(alpha=.2);ax.legend();radius_fig=picture(fig,'信赖半径变化')

fig,ax=plt.subplots(figsize=(11,5.2))
for row in pair['trials']:
    accepted=[h for h in row['history'] if 'accepted_trial' in h]
    ax.plot([h['step'] for h in accepted],[h['trials'][h['accepted_trial']]['rho'] for h in accepted],marker='o',label=labels[row['name']])
    if row['name']!='line_search':
        rejected=[(h['step'],v['rho']) for h in row['history'] for j,v in enumerate(h.get('trials',[]))
                  if 'rho' in v and j!=h.get('accepted_trial')]
        if rejected:ax.scatter(*zip(*rejected),marker='x',alpha=.35,color='gray')
ax.axhline(1,color='black',ls='--');ax.axhline(.1,color='gray',ls=':')
ax.set(xlabel='外层步骤',ylabel='实际下降 / 模型预测下降 ρ',title='线性模型对实际下降的预测质量')
ax.set_yscale('symlog',linthresh=1)
ax.grid(alpha=.2);ax.legend();rho_fig=picture(fig,'预测下降与实际下降之比')

waveforms=''
for row in pair['trials']:
    data=np.load(ROOT/(row['name']+'.npz'));power=np.sum(abs(data['output'])**2,axis=0)
    t=(np.arange(len(power))-len(power)/2)*.125
    fig,ax=plt.subplots(figsize=(11,4.5));ax.plot(t,power)
    ax.set(xlabel='时间 / ps',ylabel='输出瞬时功率 / W',title=labels[row['name']]+'：停止态输出波形')
    ax.grid(alpha=.2);waveforms+=picture(fig,row['name']+'_输出波形')

diagnostic_table=''
for row in taylor:
    full=row['taylor'][0]
    diagnostic_table+='<tr>'+''.join(f'<td>{v}</td>' for v in [row['step'],
        f"{row['epsilon_sweep'][2]['forward_relative']:.3g}",f"{row['central_linear_residual']:.4f}",
        f"{full.get('rho',float('nan')):.3g}",f"{row['gauge']['condition']:.2f}",
        f"{row['step_blocks']['field']:.3f}",f"{row['step_blocks']['population_rms']:.3g}",
        f"{row['population_feasible_alpha_limit']:.0f}"])+'</tr>'
results=''
baseline,hook=pair['trials'][:2]
conclusion=(f"同初态、同Arnoldi的12步对照：Hookstep联合残差{hook['residual']:.5g}，"
 f"比线搜索{baseline['residual']:.5g}降低{100*(1-hook['residual']/baseline['residual']):.1f}%；"
 f"映射调用减少{100*(1-hook['evaluations']/baseline['evaluations']):.1f}%。"
 f"末态规范条件数{hook['gauge']['condition']:.2f}，未触发条件数20的规范重设实验。"
 "全局化改善有直接对照证据，但12步仍未收敛，不能把这个结果解释为已越过全部平台。"
 f"Hookstep输出{hook['output_energy_nJ']:.3f} nJ，也超出0.1–0.5 nJ设计目标。")
for row in pair['trials']:
    results+='<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [labels[row['name']],status[row['status']],
        f"{row['residual']:.4g}",f"{row['relative_field_residual']:.4g}",f"{row['population_gap']:.4g}",
        f"{row['output_energy_nJ']:.5f}",row['peaks'],f"{row['seconds']:.1f}",row['evaluations']])+'</tr>'
soft_table=''
for i,row in enumerate(coarse['soft_modes'],1):
    soft_table+='<tr>'+''.join(f'<td>{v}</td>' for v in [i,f"{row['singular_value']:.4g}"]+
        [f"{100*row[k]:.4f}%" for k in ['field_fraction','population_fraction','phase_fraction','time_fraction','field_wing_fraction']])+'</tr>'
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：Newton全局化与Hookstep验证</title>
<style>body{{margin:0;background:#edf2f7;color:#172e43;font:17px/1.8 "Microsoft YaHei",sans-serif}}main{{max-width:1150px;margin:auto;padding:36px 22px}}section{{background:white;padding:30px;margin:24px 0;border-radius:10px}}h1{{font-size:32px}}h2{{font-size:25px;color:#155785}}img{{width:100%;height:auto;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ced8e2;padding:10px;text-align:left}}.scroll{{overflow:auto}}.note{{padding:16px;border-left:4px solid #d28d28;background:#fff5df}}</style><main>
<h1>原生10 MHz：Newton全局化与Hookstep验证</h1><p>固定20.42 m · CNT → OC · OC 80% · GDD +0.2 ps² · pump 50 mW · 8192点 / 0.125 ps</p>
<section><h2>1．研究问题与成对对照范围</h2><p>检验小接受步长是否由非线性模型失效、差分导数、规范或窗口造成，并比较改变步长方向的Hookstep与只缩短Newton方向的线搜索。旧LGMRES轨迹已重新执行，各步残差最大差为{replay['maximum_history_difference']:.3g}，补存原第8–11步状态和方向。</p><p>主对照两组共享同一Arnoldi、右预条件、初态、模板、缩放、网格与最多12步预算，变化仅为全局化方法。旧LGMRES与新Arnoldi不混作单因素对照。主Hookstep若出现规范条件数&gt;20，另从相同初态增加规范重设组。</p><p class="note">本轮首先检验ρ、有限信赖半径及残差下降，不把求解进展、单个局部峰或子空间诊断当作稳定锁模认证。</p></section>
<section><h2>2．第8–11步的导数与局部模型诊断</h2><div class="scroll"><table><tr><th>原步骤</th><th>ε=10⁻⁷导数相对差</th><th>中心差分线性残差</th><th>完整步ρ</th><th>规范条件数</th><th>场步范数</th><th>反转步RMS</th><th>反转可行α上限</th></tr>{diagnostic_table}</table></div><p>中心参考采用单位方向ε=10⁻⁶。完整Newton步预测下降却使真实残差显著增加；反转可行α上限远大于1，当前缩步不是N=0/1边界迫使的。场更新占主要步长。</p>{epsilon_fig}{taylor_fig}{symmetric_fig}<p>前向归一化余项在小步长区约随h下降；对称归一化余项约随h²下降。对称Jv并不使普通前向Taylor余项自动变为二阶。差分误差检查支持此处导数可用，但不是所有状态/网格的普遍保证。</p></section>
<section><h2>3．规范、窗口与投影梯度检查</h2><p>规范矩阵使用规范化行/列计算条件数，原第11步约153。规范重设只更换截面，保持场缩放、时间尺度与每圈相位/时移不变；物理残差前后应相同，不能靠改变单位制造收敛。</p><p>同一末态零填充到2.048 ns、dt不变：残差由{window['original_residual']:.10f}变为{window['wide_residual']:.10f}；范数相对变化{window['relative_norm_change']:.3g}，残差向量相对变化{window['residual_vector_relative_change']:.3g}。本例窗口效应不足以解释约10⁻²平台，但其绝对误差仍可能影响更严格稳态认证。</p><p>原第11步重建Jc：归一化投影梯度||JcᵀRc||/(||Jc|| ||Rc||)={coarse['normalized_projected_gradient']:.4f}，没有接近零的迹象。最弱右奇异向量主要在场的频谱翼；这不是Floquet模态，也不能仅据此认定物理分岔或全空间无根。</p><div class="scroll"><table><tr><th>从最小起</th><th>奇异值</th><th>场份额</th><th>反转份额</th><th>相位份额</th><th>时移份额</th><th>场频谱翼份额</th></tr>{soft_table}</table></div><p>份额按现有缩放坐标下的右奇异向量平方范数计算；频谱翼指|FFT索引|&gt;64，是场份额的子集。</p></section>
<section><h2>4．右预条件Hookstep的步长约束</h2><p>构造JZ=VH，Z=MV为经过右预条件的真实状态方向。求解 min||βe₁−Hy||，约束||DZy||≤Δ；通过DZ的SVD白化小型约束问题，不能用||y||≤Δ替代。</p><p>D=I只针对已经缩放的坐标：场除以初始场范数、反转按RMS、相位以rad计、时移除以固定初始时间尺度。这与残差权重是不同概念。拒绝步复用同一Krylov空间；按ρ=实际下降/预测下降调整半径，不裁剪反转。</p><p>主对照固定规范；条件触发的规范重设是独立因素。方法参考<a href="https://chaosbook.org/library/ViswanathJFM07.pdf">Viswanath (JFM, 2007)的Newton–Krylov与受约束hookstep</a>，不迁移其流体算例的收敛保证。</p></section>
<section><h2>5．完整全局化对照结果</h2><div class="scroll"><table><tr><th>方法</th><th>停止状态</th><th>联合残差</th><th>相对场残差</th><th>最大反转差</th><th>输出/nJ</th><th>局部峰数</th><th>秒</th><th>映射数</th></tr>{results}</table></div>{convergence_fig}{radius_fig}{rho_fig}<p>Hookstep的接受门槛本来就要求ρ&gt;0.1，因此不能把“接受步ρ为正”单独当作机制证据。灰色叉号保留拒绝步；应结合ρ接近1的程度、半径是否塌缩、整体残差与成本判断。</p>{waveforms}</section>
<section><h2>6．数值验收与结论边界</h2><p>测试包含100倍右预条件放大下的真实步长约束、与独立约束优化器核对、已知线性解、Rosenbrock非线性问题、规范重设的物理残差不变性及已有模型回归。默认旧求解入口不被实验方法替换。</p><p>没有通过非零局域稳态、时频边界和细网格检验的状态，不计算Floquet认证或稳定支路。有限步数未收敛、投影梯度或奇异向量均不足以证明物理上不存在目标单脉冲。原报告保持不变。</p></section></main></html>'''
download='data:application/json;base64,'+base64.b64encode((ROOT/'hookstep_pair.json').read_bytes()).decode()
body=body.replace('</main>',f'<p><a download="hookstep_pair.json" href="{download}">下载完整成对迭代记录</a></p></main>')
body=body.replace('<h2>5．完整全局化对照结果</h2>', '<h2>5．完整全局化对照结果</h2><p class="note">'+conclusion+'</p>')
body=body.replace('<h2>6．数值验收与结论边界</h2>', '<h2>6．数值验收与结论边界</h2>'
 f"<p>Hookstep末态时间边缘能量份额{hook['time_edge']:.3g}、频谱边缘份额{hook['spectral_edge']:.3g}；这些仅是当前网格诊断，不代替细网格收敛检验。39项CPU回归测试通过。</p>")
path=ROOT/'原生10MHz_Newton全局化与Hookstep验证.html';path.write_text(body,encoding='utf8');print(path)
