"""Window and projected soft-mode evidence, followed by bounded continuation."""
import base64,html,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT,PROJECT
from steady_preconditioner import spectral_coordinates

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':12})
read=lambda name:json.loads((ROOT/name).read_text(encoding='utf8'))
w=read('endpoint_window.json');soft=read('endpoint_soft_modes.json');run=read('tail_continuation.json')
old=np.load(PROJECT/'results/steady_hookstep_20260919/hookstep.npz')
final=np.load(ROOT/'tail_continuation.npz');hist=run['history']

def picture(name,draw):
    fig,ax=plt.subplots(figsize=(11,5));draw(ax);ax.grid(alpha=.2)
    path=ROOT/(name+'.png');fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
    return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'

def trace(ax):
    for label,a in [('原Hookstep末态',old['a']),('宽窗续算末态',final['a'])]:
        t=(np.arange(a.shape[-1])-a.shape[-1]/2)*.125;p=np.sum(abs(a)**2,axis=0)
        ax.semilogy(t,np.maximum(p/p.max(),1e-18),label=label)
    ax.axvline(-512,color='gray',ls='--');ax.axvline(512,color='gray',ls='--')
    ax.set(xlabel='时间 / ps',ylabel='归一化腔内功率',title='时间尾部与原窗口边界',ylim=(1e-15,2));ax.legend()
tail_fig=picture('时间尾部与窗口',trace)
res=np.load(ROOT/'endpoint_residuals.npz');n=old['a'].shape[-1]
def residual_trace(ax):
    for label,r,nn in [('1.024 ns窗口',res['r'],n),('2.048 ns窗口',res['rw'],2*n)]:
        z=(r[:2*nn]+1j*r[2*nn:4*nn]).reshape(2,nn)
        ax.semilogy((np.arange(nn)-nn/2)*.125,np.maximum(np.sum(abs(z)**2,axis=0),1e-24),label=label)
    ax.set(xlabel='时间 / ps',ylabel='缩放场残差的局部平方和',title='相同物理状态在两个窗口中的一圈闭合误差');ax.legend()
res_fig=picture('同态残差空间分布',residual_trace)
_,lift,_=spectral_coordinates(n,len(old['pop']),384)
modes=np.load(ROOT/'original_soft_modes.npz')['vectors'][::-1]
fields=[]
for v in modes:
    d=lift(v);fields.append((d[:2*n]+1j*d[2*n:4*n]).reshape(2,n))
def mode_time(ax):
    for j,z in enumerate(fields):ax.semilogy((np.arange(n)-n/2)*.125,np.maximum(np.sum(abs(z)**2,axis=0),1e-20),label=f'第{j+1}弱方向')
    ax.set(xlabel='时间 / ps',ylabel='单位奇异向量的场份额 / 采样点',title='投影软方向的时间分布');ax.legend()
def mode_frequency(ax):
    for j,z in enumerate(fields):
        ax.semilogy(np.fft.fftshift(np.fft.fftfreq(n,.125))*1000,np.maximum(np.fft.fftshift(np.sum(abs(np.fft.fft(z,norm='ortho'))**2,axis=0)),1e-20),label=f'第{j+1}弱方向')
    ax.axvline(-62.5,color='gray',ls='--');ax.axvline(62.5,color='gray',ls='--')
    ax.set(xlabel='相对中心频率 / GHz',ylabel='单位奇异向量的频谱份额 / 频点',title='投影软方向的频谱分布',xlim=(-400,400),ylim=(1e-12,1));ax.legend()
mode_t=picture('软方向时间分布',mode_time);mode_f=picture('软方向频谱分布',mode_frequency)
def convergence(ax):
    ax.semilogy([h['step'] for h in hist],[h['residual'] for h in hist],marker='o')
    ax.axhline(1e-7,color='gray',ls='--');ax.set(xlabel='宽窗追加步骤',ylabel='联合缩放残差',title='保持物理参数的宽窗Hookstep续算')
conv=picture('宽窗续算残差',convergence)
def local_convergence(ax):
    ax.plot([h['step'] for h in hist],[h['residual']/hist[0]['residual'] for h in hist],marker='o')
    ax.set(xlabel='宽窗追加步骤',ylabel='残差 / 本轮初始残差',title='宽窗续算下降幅度（局部尺度）')
detail=picture('宽窗续算下降幅度',local_convergence)
def radius(ax):
    ax.semilogy([h['step'] for h in hist],[h['radius'] for h in hist],marker='o');ax.set(xlabel='宽窗追加步骤',ylabel='下一步信赖半径',title='续算信赖半径变化')
rad=picture('宽窗续算信赖半径',radius)
def rho_plot(ax):
    accepted=[h for h in hist if 'accepted_trial' in h]
    ax.plot([h['step'] for h in accepted],[h['trials'][h['accepted_trial']]['rho'] for h in accepted],marker='o',label='接受步ρ')
    rejected=[(h['step'],v['rho']) for h in hist for j,v in enumerate(h.get('trials',[])) if 'rho' in v and j!=h.get('accepted_trial')]
    if rejected:ax.scatter(*zip(*rejected),marker='x',color='gray',label='拒绝步ρ')
    ax.set_yscale('symlog',linthresh=1);ax.set(xlabel='宽窗追加步骤',ylabel='实际 / 预测下降',title='宽窗续算的模型预测质量');ax.legend()
rho=picture('宽窗续算预测质量',rho_plot)
def step_plot(ax):
    rows=[h for h in hist if 'accepted_trial' in h]
    ax.semilogy([h['step'] for h in rows],[h['newton_blocks']['total'] for h in rows],marker='o',label='未约束Newton方向')
    ax.semilogy([h['step'] for h in rows],[h['trials'][h['accepted_trial']]['step_blocks']['total'] for h in rows],marker='o',label='实际接受步')
    ax.set(xlabel='宽窗追加步骤',ylabel='固定缩放坐标中的范数',title='Newton方向与实际接受步长');ax.legend()
steps=picture('方向范数与接受步长',step_plot)
def output(ax):
    p=np.sum(abs(final['output'])**2,axis=0);t=(np.arange(len(p))-len(p)/2)*.125
    ax.plot(t,p);ax.set(xlabel='时间 / ps',ylabel='输出瞬时功率 / W',title='宽窗续算停止态输出（不代表已认证稳态）')
out=picture('宽窗停止态输出',output)

table=''
for i,(a,b) in enumerate(zip(soft[0]['modes'],soft[1]['modes']),1):
    values=[i,f"{a['singular_value']:.6g}",f"{b['singular_value']:.6g}",f"{a['original_full_Jv_norm']:.6g}",
        f"{a['wide_full_Jv_norm']:.6g}",f"{100*a['field_wing_fraction']:.3f}%",f"{100*a['time_outside_100ps_fraction']:.5f}%"]
    table+='<tr>'+''.join('<td>'+str(v)+'</td>' for v in values)+'</tr>'
stop_labels={'iteration_budget_reached':'达到25步预算','progress_stagnated':'触发五步进展停止条件','globalization_stalled':'试探步未能接受','radius_floor_reached':'半径达到数值下限','residual_converged':'残差通过门槛，仍需网格与物理验收'}
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：末态窗口、软方向与续算验证</title>
<style>body{{margin:0;background:#edf2f7;color:#172e43;font:17px/1.8 "Microsoft YaHei",sans-serif}}main{{max-width:1150px;margin:auto;padding:36px 22px}}section{{background:white;padding:30px;margin:24px 0;border-radius:10px}}h1{{font-size:32px}}h2{{font-size:25px;color:#155785}}img{{width:100%;height:auto;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ced8e2;padding:10px}}.scroll{{overflow:auto}}.note{{padding:16px;border-left:4px solid #d28d28;background:#fff5df}}code{{overflow-wrap:anywhere}}</style><main>
<h1>原生10 MHz：末态窗口、软方向与续算验证</h1><p>固定20.42 m · CNT → OC · OC 80% · GDD +0.2 ps² · pump 50 mW · dt 0.125 ps</p>
<section><h2>1．本轮要区分的数值问题</h2><p>上一轮Hookstep末态联合残差0.004603，时间边缘能量增至6.77×10⁻⁷。本轮先检查这个具体状态的窗口敏感性，再检验软方向是否随宽窗消失，最后继续宽窗求解。能量目标仍是0.1–0.5 nJ；当前约1.8 nJ状态仅作为非零根的求解入口，尚不是已确认的稳态锚点。</p><p class="note">窗口范数相近不代表残差向量相近；投影奇异值小不等于完整雅可比具有同样小的奇异值；固定点求解进展不等于物理稳定。</p></section>
<section><h2>2．同一末态的时间窗口复核</h2><p>将8192点场中央嵌入16384点，外侧补零；反转、每圈相位与时移、场范数、时间尺度及规范模板全部保留。比较R宽与零填充后的R原，而不是重新优化状态。</p><div class="scroll"><table><tr><th>量</th><th>结果</th></tr><tr><td>原窗口 / 宽窗口残差范数</td><td>{w['original_residual']:.9f} / {w['wide_residual']:.9f}</td></tr><tr><td>范数相对变化</td><td>{100*w['relative_norm_change']:.5f}%</td></tr><tr><td>原中心区域的向量相对差</td><td>{100*w['center_vector_relative_change']:.4f}%</td></tr><tr><td>含外部区域的完整向量相对差</td><td>{100*w['full_vector_relative_change']:.4f}%</td></tr><tr><td>新增外部区域的残差范数</td><td>{w['outer_residual_norm']:.5g}</td></tr></table></div><p>向量差约1.22%，足以要求在宽窗重新验证后期求解，但不足以认定整个0.0046残差由边界造成。补零后的输入边缘自动变为零，不是边界收敛证据；这里同时检查传播后的残差。尚未排除严格10⁻⁷认证中的窗口误差。</p>{res_fig}</section>
<section><h2>3．末态残差的场与反转组成</h2><p>同一缩放坐标中，场块范数{w['original_blocks']['field']:.6g}，反转RMS残差{w['original_blocks']['population']:.6g}，规范残差{w['original_blocks']['gauge']:.6g}。场块占联合残差平方的{100*(w['original_blocks']['field']/w['original_residual'])**2:.2f}%。这支持“当前代数残差以场闭合误差为主”，但不能把反转耦合或真实慢增益动力学排除。</p></section>
<section><h2>4．相同扰动子空间中的软方向对照</h2><p>原局域Fourier子空间为±375 GHz、3093维。宽窗对照使用同一组零嵌入扰动，并将输出限制回相同中心子空间：Jc宽=P J宽 E。这样没有把窗口加倍与频带减半混在一起，但也没有覆盖新增外部自由度。</p><div class="scroll"><table><tr><th>弱方向排序</th><th>原σ</th><th>宽窗投影σ</th><th>原完整‖Jv‖</th><th>宽完整‖Jv‖</th><th>频谱翼份额</th><th>|t|&gt;100 ps份额</th></tr>{table}</table></div><p>频谱翼定义为|f|&gt;62.5 GHz，各份额以完整单位向量平方范数为分母。最弱方向的频谱翼约68.3%，时间上主要位于脉冲附近，不能沿用旧状态“99%频谱翼”的解释。宽窗投影σ几乎不变，没有看到边界加宽使该投影软方向消失。</p><p>完整Jv使用独立中心差分；最弱方向完整响应约为投影σ的4.5倍，说明投影丢弃了部分响应。以上结果不证明全空间近奇异、物理折叠或Floquet临界模态。归一化投影梯度为{soft[0]['normalized_projected_gradient']:.4f}，也不足以声称到达驻点。</p>{mode_t}{mode_f}</section>
<section><h2>5．宽窗Hookstep续算与进展判据</h2><p>全场残差和Krylov差分均在2.048 ns宽窗计算。为了保留原物理频带且控制构建成本，预条件逆使用原1.024 ns中心区域的3093维子空间；外部场用−I近似。新增外部自由度仍在完整Krylov求解中，非线性映射不截断。与上一轮并非“只改变窗口”的性能对照。</p><p>保留末次半径0.0125，最多追加25步。最近五个接受步下降&gt;10%标记进展、2–10%标记缓慢；低于2%且其中至少四个ρ&gt;0.5时停止并诊断。这是操作判据，不是平坦谷或无根定理。</p><p class="note">停止原因：{stop_labels.get(run['status'],run['status'])}；实际追加{hist[-1]['step']}步。联合残差{run['residual']:.6g}，相对场残差{run['relative_field_residual']:.6g}，最大反转差{run['population_gap']:.6g}。耗时{run['seconds']:.1f}秒，映射{run['evaluations']}次。</p>{conv}{rad}{rho}{tail_fig}{out}<p>输出{run['output_energy_nJ']:.6f} nJ、局部峰数{run['peaks']}；时间边缘能量份额{run['time_edge']:.4g}、频谱边缘{run['spectral_edge']:.4g}。局部峰数不是单脉冲稳定认证。</p></section>
<section><h2>6．PTC与后续方法的适用前提</h2><p>当前残差场块是R场=Φ(A)−A，反转块是R反转=N−N平衡，符号相反。若构造人工松弛Ȧ=R场、Ṅ=−R反转，半隐式线性步对应(J+μS)s=−R，其中S场=−I、S反转=+I；规范行保持代数约束，不给相位/时移随意添加质量项。</p><p>因此不能直接使用统一的J+μI并称为稳定化。例如Φ(a)=0.5a时R=−0.5a，μ=1的正移位会给出s=+a，反而放大误差。这个人工流不是真实EDF慢动力学；稳定性仍需检验。非正规矩阵中，特征值移位也不保证小奇异值单调增大。</p><p>PTC利用合适的演化结构计算稳态，参见<a href="https://doi.org/10.1137/S0036142996304796">Kelley与Keyes，SIAM J. Numer. Anal. (1998)</a>。若需切换，将以当前停止状态作受控移位测试，先检查方向、真实残差下降及边界，不宣称方法必然优于Hookstep。Multiple shooting需定义分段场、同一EDF反转及循环匹配；不能仅因原映射非线性就保证分段后更易收敛。</p></section>
<section><h2>7．结论边界与可复现记录</h2><p>已完成当前末态宽窗、同扰动子空间软方向对照及带停止逻辑的续算。窗口影响不能忽略，现有软方向却未因宽窗而消失；它也不是纯尾部向量。仍需把求解器投影几何与完整物理分支性质分开。</p><p>只有非零局域根、严格残差、时频边界与细网格一致性均通过后，才能把状态交给Floquet或泵浦支路延拓。旧报告未改写，本轮结果不回填为历史认证。</p></section></main></html>'''
download='data:application/json;base64,'+base64.b64encode((ROOT/'tail_continuation.json').read_bytes()).decode()
body=body.replace('</main>',f'<a download="tail_continuation.json" href="{download}">下载完整续算记录</a></main>')
body=body.replace(rho,rho+steps)
body=body.replace(conv,conv+detail)
linear_table=''
for h in [v for v in hist if 'true_newton_linear_residual' in v][-6:]:
    values=[h['step'],h['krylov_dimension'],f"{h['true_newton_linear_residual']:.4g}",
            f"{h['newton_blocks']['total']:.4g}",f"{h['coarse_condition']:.4g}",h['precondition_rebuilt']]
    linear_table+='<tr>'+''.join('<td>'+str(v)+'</td>' for v in values)+'</tr>'
body=body.replace('<h2>6．PTC与后续方法的适用前提</h2>', '<h2>6．PTC与后续方法的适用前提</h2>'
    '<p>下表先检查本轮后期线性子问题，不能沿用上一轮“线性精度已足够”的判断。预条件条件数仅在重建时更新，其他行沿用旧矩阵。</p>'
    '<div class="scroll"><table><tr><th>追加步</th><th>Krylov维数</th><th>真实Newton线性相对残差</th><th>Newton方向范数</th><th>预条件投影条件数</th><th>本步重建</th></tr>'+linear_table+'</table></div>')
if (ROOT/'final_window_check.json').exists():
    last=read('final_window_check.json')
    body=body.replace('<h2>7．结论边界与可复现记录</h2>', '<h2>7．结论边界与可复现记录</h2>'
        f"<p>新的续算末态再次加宽到4.096 ns：范数相对变化{100*last['relative_norm_change']:.5f}%，完整残差向量相对差{100*last['full_vector_relative_change']:.4f}%，外部残差范数{last['outer_residual_norm']:.4g}。此检查与原末态的宽窗测试分开记录。</p>")
if (ROOT/'ptc_direction_pilot.json').exists():
    pilot=read('ptc_direction_pilot.json');ptc_table=''
    def ptc_plot(ax):
        for row in pilot['rows']:
            trials=[t for t in row['trials'] if 'residual' in t]
            ax.semilogx([t['alpha']*row['direction_blocks']['total'] for t in trials],
                [t['residual']/pilot['initial_residual'] for t in trials],marker='o',label=f"μ={row['mu']:g}")
        ax.axhline(1,color='gray',ls='--');ax.set(xlabel='实际缩放步长 ‖s‖',ylabel='试探残差 / 初始残差',title='同状态移位方向的真实残差变化');ax.legend()
    ptc_fig=picture('PTC移位方向真实残差',ptc_plot)
    for row in pilot['rows']:
        best=min([t for t in row['trials'] if 'residual' in t],key=lambda t:t['residual'])
        values=[row['mu'],f"{row['direction_blocks']['total']:.4g}",f"{row['true_shifted_relative']:.4g}",best['cap'],f"{best['residual']:.7g}"]
        ptc_table+='<tr>'+''.join('<td>'+str(v)+'</td>' for v in values)+'</tr>'
    body=body.replace('<h2>6．PTC与后续方法的适用前提</h2>', '<h2>6．PTC与后续方法的适用前提</h2>'
        '<p>已触发续算停止条件，进行同状态PTC方向试验。各μ使用相同基态、相同步长上限集合；μ=0为Newton方向对照。表列四个上限中最好的一次试探，并非多步PTC求解。</p>'
        '<div class="scroll"><table><tr><th>μ</th><th>方向范数</th><th>真实移位线性残差</th><th>最佳步长上限</th><th>试探后原方程残差</th></tr>'+ptc_table+'</table></div>'+ptc_fig)
    best=min((t for row in pilot['rows'] for t in row['trials'] if 'residual' in t),key=lambda t:t['residual'])
    body=body.replace('<h2>7．结论边界与可复现记录</h2>', '<h2>7．结论边界与可复现记录</h2>'
        f"<p class='note'>PTC方向试验的最好试探残差为{best['residual']:.7g}，单步下降{100*(1-best['residual']/pilot['initial_residual']):.3f}%。μ=0.1、1虽然压小了方向，却使测试步的原方程残差增加。现有证据不支持“增大移位即可越过平台”。这只是同态方向对照，没有运行多步PTC，也没有与同态Hookstep方向作性能配对，不能宣称优于或劣于Hookstep整体方法。</p>")
else:
    body=body.replace('<h2>6．PTC与后续方法的适用前提</h2>', '<h2>6．PTC与后续方法的适用前提</h2><p class="note">本轮未触发停滞切换条件，未执行PTC轨迹或multiple shooting。以下为实现前的数学核对，不能读作已完成的性能结果。</p>')
if (ROOT/'late_derivative_check.json').exists():
    derivative=read('late_derivative_check.json')
    body=body.replace('<h2>6．PTC与后续方法的适用前提</h2>', '<h2>6．PTC与后续方法的适用前提</h2>'
        f"<p>独立导数复核（追加第{derivative['step']}步）：方向范数{derivative['direction_norm']:.3f}，单位方向前向ε=10⁻⁷与中心ε=10⁻⁶差分的相对差{derivative['forward_central_relative_difference']:.4g}；中心差分复核的Newton线性相对残差{derivative['central_linear_relative_residual']:.4g}。这只验证该方向，不证明所有软方向均可靠。</p>")
path=ROOT/'原生10MHz_末态窗口与软方向续算验证.html';path.write_text(body,encoding='utf8');print(path)
