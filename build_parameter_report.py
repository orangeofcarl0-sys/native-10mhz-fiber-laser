"""Human-readable evidence for parameter release; no fold claim from non-roots."""
import base64,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':12})
read=lambda name:json.loads((ROOT/name).read_text(encoding='utf8'))
g=read('parameter_geometry.json');run=read('bordered_pilot.json');hist=run['history']
labels={'pump':'泵浦功率','gdd':'净色散','oc':'输出耦合','psat':'CNT饱和功率'}
def image(name,draw):
    fig,ax=plt.subplots(figsize=(11,5));draw(ax);ax.grid(alpha=.2)
    path=ROOT/(name+'.png');fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
    return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'
def gradient(ax):
    rows=g['gradient_trials'];ax.semilogx([v['step'] for v in rows],[100*(v['residual']/g['residual']-1) for v in rows],marker='o')
    ax.set_yscale('symlog',linthresh=3)
    ax.axhline(0,color='gray',ls='--');ax.set(xlabel='完整merit子空间负梯度方向的步长',ylabel='相对起点的残差变化 / %',title='梯度方向步长与完整残差变化')
gradient_fig=image('完整残差梯度方向验证',gradient)
def singular(ax):
    rows=g['modes'];ax.semilogy([v['index'] for v in rows],[v['sigma'] for v in rows],marker='o',label='投影奇异值')
    if 'full_Jv_norm' in rows[0]:ax.semilogy([v['index'] for v in rows],[v['full_Jv_norm'] for v in rows],marker='s',label='完整‖Jv‖')
    ax.set(xlabel='由最弱起的方向序号',ylabel='固定缩放坐标中的响应大小',title='末态最弱20个方向：投影与完整响应');ax.legend()
singular_fig=image('最弱20方向的雅可比响应',singular)
def coefficients(ax):
    rows=g['modes'];ax.semilogy([v['index'] for v in rows],[abs(v['newton_coefficient']) for v in rows],marker='o')
    ax.set(xlabel='由最弱起的方向序号',ylabel=r'$|u_i^T R_c|/\sigma_i$',title='投影Newton方向中的软模系数')
coefficient_fig=image('软模残差系数放大',coefficients)
def coupling(ax):
    rows=g['parameters'];ax.bar([labels[v['name']] for v in rows],[v['eta_projected'] for v in rows])
    ax.set(ylabel=r'$|u_{\min}^T R_{q,c}|/\|R_{q,c}\|$',title='四个参数与最弱左方向的归一化耦合')
coupling_fig=image('参数归一化软方向耦合',coupling)
def conditioning(ax):
    rows=g['parameters'];ax.bar([labels[v['name']] for v in rows],[v['bordered_condition'] for v in rows])
    ax.axhline(g['condition'],color='gray',ls='--',label='固定参数投影矩阵');ax.set_yscale('log')
    ax.set(ylabel='矩阵条件数（依赖所列参数尺度）',title='加入参数列和截面约束后的投影条件数');ax.legend()
condition_fig=image('参数边框矩阵条件对照',conditioning)
def convergence(ax):
    ax.semilogy([h['step'] for h in hist],[h['residual'] for h in hist],marker='o')
    ax.axhline(1e-7,color='gray',ls='--');ax.set(xlabel='增广求解步骤',ylabel='增广残差',title='泵浦增广Newton–Hookstep求解')
conv=image('泵浦增广求解残差',convergence)
def detail(ax):
    ax.plot([h['step'] for h in hist],[h['residual']/hist[0]['residual'] for h in hist],marker='o')
    ax.set(xlabel='增广求解步骤',ylabel='增广残差 / 初值残差',title='增广求解实际下降幅度')
detail_fig=image('泵浦增广残差下降幅度',detail)
def rho(ax):
    rows=[h for h in hist if 'accepted_trial' in h]
    ax.plot([h['step'] for h in rows],[h['trials'][h['accepted_trial']]['rho'] for h in rows],marker='o',label='接受步ρ')
    rejected=[(h['step'],v['rho']) for h in hist for j,v in enumerate(h.get('trials',[])) if 'rho' in v and j!=h.get('accepted_trial')]
    if rejected:ax.scatter(*zip(*rejected),marker='x',color='gray',label='拒绝步ρ')
    ax.set_yscale('symlog',linthresh=1);ax.set(xlabel='增广求解步骤',ylabel='实际 / 预测下降',title='参数释放后的局部模型质量');ax.legend()
rho_fig=image('增广求解预测质量',rho)
state=np.load(ROOT/'bordered_pilot.npz')
def output(ax):
    p=np.sum(abs(state['output'])**2,axis=0);t=(np.arange(len(p))-len(p)/2)*.125
    ax.plot(t,p);ax.set(xlabel='时间 / ps',ylabel='输出瞬时功率 / W',title='增广试验停止态输出（非稳定性认证）')
out=image('增广停止态输出',output)
mode_table=''
for row in g['modes']:
    mode_table+='<tr>'+''.join('<td>'+v+'</td>' for v in [str(row['index']),f"{row['sigma']:.5g}",f"{abs(row['left_residual']):.5g}",f"{abs(row['newton_coefficient']):.5g}",f"{row.get('full_Jv_norm',float('nan')):.5g}"])+'</tr>'
parameter_table=''
for row in g['parameters']:
    value=row['base']+row['scale']*row['linear_parameter_change']
    values=[labels[row['name']],f"{row['scale']:g} {row['unit']}",f"{row['halving_relative_difference']:.2g}",f"{row['eta_projected']:.5f}",f"{row['left_soft_coupling']:.5g}",f"{row['bordered_condition']:.5g}",f"{value:.5g} {row['unit']}"]
    parameter_table+='<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>'
status={'progress_stagnated':'触发五步进展停止条件','iteration_budget_reached':'达到20步预算','globalization_stalled':'试探步未能接受','radius_floor_reached':'半径达到下限','residual_converged':'增广残差通过门槛，仍需物理与网格验收'}
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生10MHz：参数耦合与增广求根验证</title>
<style>body{{margin:0;background:#edf2f7;color:#172e43;font:17px/1.8 "Microsoft YaHei",sans-serif}}main{{max-width:1150px;margin:auto;padding:36px 22px}}section{{background:white;padding:30px;margin:24px 0;border-radius:10px}}h1{{font-size:32px}}h2{{font-size:25px;color:#155785}}img{{width:100%;height:auto;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ced8e2;padding:10px}}.scroll{{overflow:auto}}.note{{padding:16px;border-left:4px solid #d28d28;background:#fff5df}}</style><main>
<h1>原生10 MHz：参数耦合与增广求根验证</h1><p>固定20.42 m · CNT → OC · 起点pump 50 mW / OC 80% / GDD +0.2 ps² · 2.048 ns窗口 / dt 0.125 ps</p>
<section><h2>1．本轮问题与证据边界</h2><p>在残差0.003782的同一末态，检验是否还存在一阶下降方向，以及释放物理参数能否改善局部求根。先重建投影矩阵、检查左右弱方向及参数列，再作有界的泵浦联合求解。</p><p class="note">当前状态不是精确根。软方向、参数耦合或带边框条件数改善都不能单独证明fold、无根或稳定分支。此次为截面约束的局部求根，不是已经开始的伪弧长延拓。</p></section>
<section><h2>2．残差组成与完整目标的一阶下降方向</h2><p>按同一缩放坐标计算，场块占联合残差平方的{100*g['field_block_fraction']:.3f}%。不能用“相对场残差/联合残差”的平方替代该比例，因为前者按当前场范数归一化，后者按固定初始尺度。</p><p>投影目标的归一化梯度为{g['normalized_projected_gradient']:.5f}。更直接地，保留完整输出列B=J E，计算g完整子空间=BᵀR，范数{g['full_objective_subspace_gradient_norm']:.7f}。沿其单位负方向，中心差分验证的merit导数为{g['full_gradient_directional_derivative']:.7f}，预测为{g['full_gradient_directional_prediction']:.7f}，两者一致。</p><p>步长10⁻⁴使完整残差降到0.003705，直接表明此点还存在可用的一阶下降方向。更大梯度步会增大残差，说明需要局部步长控制。该结果不回答根是否存在，但否定了“此次工程停止就等于已到非零残差极小”的解释。</p>{gradient_fig}</section>
<section><h2>3．最弱20个方向的残差投影与放大</h2><p>局域中心1.024 ns、±375 GHz的3093维投影矩阵，在完整2.048 ns传播器上差分构建。最小奇异值{g['sigma_min']:.6g}，条件数{g['condition']:.6g}。若Jc=UΣVᵀ，则投影Newton系数为−(uᵢᵀRc)/σᵢ。</p><p>最弱20个方向占投影Newton步范数的{100*g['soft20_newton_norm_fraction']:.6f}%。这是投影逆对残差的放大，不等于完整系统的Newton步。下表同时列出独立中心差分的完整‖Jv‖，避免把投影奇异值当成完整奇异值。</p>{singular_fig}{coefficient_fig}<div class="scroll"><table><tr><th>序号</th><th>σᵢ</th><th>|uᵢᵀRc|</th><th>|cᵢ/σᵢ|</th><th>完整‖Jvᵢ‖</th></tr>{mode_table}</table></div></section>
<section><h2>4．四个物理参数的导数与耦合</h2><p>用q=(参数−基准)/尺度作差分，q步长10⁻³与5×10⁻⁴核对。表中η=|u最弱ᵀRq,c|/‖Rq,c‖，是投影方向的归一化耦合，仍不是完整系统的fold横截条件。不同单位的裸导数不直接排序；边框条件数则依赖这里明确列出的尺度。</p><p>泵浦改变EDF速率与反转平衡；GDD改变SMF/NDF长度分配并保持总腔长；OC改变输出与腔内回馈比例；CNT饱和功率定义为Esat/τ，固定τ=0.5 ps，因此40 W对应20 pJ。不是凭空增加一个与当前CNT模型无关的参数。</p><div class="scroll"><table><tr><th>参数</th><th>q=1尺度</th><th>步长减半差</th><th>η</th><th>uᵀRq,c</th><th>边框条件数</th><th>完整线性步外推值（非解）</th></tr>{parameter_table}</table></div>{coupling_fig}{condition_fig}<p>泵浦η最大，因此选为首个联合求根参数；OC的缩放边框条件数更低，不能把泵浦称作普遍最优参数。四个完整线性参数步均超出预设的±2尺度范围，尤其泵浦外推约13.4 mW；这不支持“50 mW只需微调一点就有根”的断言，也不能据线性外推排除非线性局部根。</p></section>
<section><h2>5．泵浦与光场的有界联合求解</h2><p>未知量Y=(X,q)，附加约束v最弱ᵀ(X−X起点)=0。其作用是固定一个截面，让参数变化代替该方向的自由漂移；截面本身是本次试验选择，不保证穿过附近真实根。</p><p>泵浦限定40–60 mW，q∈[−2,2]；最多20步，五步进展停止规则与前轮一致。完整残差、规范和Krylov均在宽窗上计算，局域带边框预条件维数3094；场和反转单位保持原值，q以5 mW为一步尺度。</p><p class="note">停止原因：{status.get(run['status'],run['status'])}；实际步骤{hist[-1]['step']}。最终泵浦{1000*run['value']:.6f} mW，原方程残差{run['physical_residual']:.7g}，增广残差{run['augmented_residual']:.7g}，截面残差{run['hyperplane_residual']:.4g}。耗时{run['seconds']:.1f}秒，映射调用{run['evaluations']}次。</p>{conv}{detail_fig}{rho_fig}{out}<p>停止态输出{run['output_energy_nJ']:.6f} nJ，局部峰数{run['peaks']}；时间边缘能量{run['time_edge']:.4g}，频谱边缘{run['spectral_edge']:.4g}。未把这些数值当作稳定单脉冲认证。</p></section>
<section><h2>6．对分支假说的当前判断</h2><p>当前点仍有完整merit的一阶下降方向，不能称为已经到达非零局部极小。参数列可以显著改善投影边框条件，但不自动使非线性增广系统收敛。这里不能由有耦合推出存在simple fold，也不能由有限预算、参数边界或所选截面上的失败推出附近无根。</p><p>伪弧长方法对规则点和simple fold的增广雅可比有相应非奇异性结果，参见<a href="https://epubs.siam.org/doi/10.1137/060654384">Dickson等：Condition Estimates for Pseudo-Arclength Continuation</a>。该结果有分支与非退化条件，不能直接套在尚未找到根的本例上。</p><p>若取得精确非零根，还需细网格、窗口及原物理映射验收后再构造分支切向量。若局部参数增广未成功，下一项判别应区分截面/参数范围限制、Krylov子空间遗漏下降方向与single shooting条件；不因本轮有限试验直接跳到“物理无解”。</p></section>
<section><h2>7．复现与验收记录</h2><p>输入沿用上一轮tail_continuation.npz。参数差分、47项CPU回归与已知简单折叠边框例均检查；历史报告不覆盖。结果保存梯度、最弱20模、参数列、完整增广迭代和停止态。大型投影矩阵只是本机重建缓存，可由脚本重新生成。</p></section></main></html>'''
download='data:application/json;base64,'+base64.b64encode((ROOT/'bordered_pilot.json').read_bytes()).decode()
body=body.replace('</main>',f'<a download="bordered_pilot.json" href="{download}">下载完整增广求解记录</a></main>')
if 'full_Jv_norm' in g['modes'][0]:
    first=g['modes'][0]
    body=body.replace('<h2>3．最弱20个方向的残差投影与放大</h2>', '<h2>3．最弱20个方向的残差投影与放大</h2>'
        f"<p class='note'>最弱投影方向放回完整系统后，‖Jv‖={first['full_Jv_norm']:.6g}，是投影σ的{first['full_to_projected_ratio']:.1f}倍。当前投影近零性不能直接用作完整物理Jacobian近奇异或fold的证据。参数耦合η同样仅针对投影左方向。独立中心差分的投影响应为{first['central_projected_Jv_norm']:.6g}，投影之外响应为{first['outside_projected_response_norm']:.6g}；差异主要来自被投影舍弃的输出分量。</p>")
if (ROOT/'bordered_endpoint_check.json').exists():
    check=read('bordered_endpoint_check.json')
    note=(f"<p>末态按实际泵浦重新计算，原方程残差一致；加宽到4.096 ns后完整残差向量相对差{100*check['full_vector_relative_change']:.4f}%。"
          f"反转范围{check['population_min']:.4f}–{check['population_max']:.4f}，距离参数边界的q余量{check['parameter_bound_margin_scaled']:.4g}。</p>")
    if check['parameter_bound_margin_scaled']<.1:
        note+='<p class="note">末态接近预设参数范围边界。本实现通过拒绝越界试探并缩小整体步长处理边界，不是完整的箱约束最小二乘求解器；因此不能据停止判断范围内已无其他可行下降方向，更不能证明附近无根。扩大范围、固定边界参数重新求解或更换截面属于另外的受控实验。</p>'
    body=body.replace('<h2>6．对分支假说的当前判断</h2>','<h2>6．对分支假说的当前判断</h2>'+note)
if (ROOT/'bordered_expanded.json').exists():
    expanded=read('bordered_expanded.json');eh=expanded['history']
    expanded_status='达到25步预算' if expanded['status']=='iteration_budget_reached' else status.get(expanded['status'],expanded['status'])
    late=[h for h in eh if 'true_newton_linear_residual' in h][-5:]
    linear_values=[h['true_newton_linear_residual'] for h in late]
    linear_note=(f"<p>末五个求解步的真实线性相对残差范围{min(linear_values):.4g}–{max(linear_values):.4g}，"
                 f"Krylov维数为{[h['krylov_dimension'] for h in late]}。本次末段均达到120维上限且未达3%目标，线性求解限制已重新出现，不能沿用先前状态的结论。</p>")
    def expanded_curve(ax):
        ax.semilogy([h['step'] for h in eh],[h['residual'] for h in eh],marker='o')
        ax.set(xlabel='扩展范围后的追加步骤',ylabel='增广残差',title='5–60 mW范围的独立续算')
    def pump_curve(ax):
        ax.plot([h['step'] for h in expanded['parameter_trace']],[1000*h['pump_W'] for h in expanded['parameter_trace']],marker='o')
        ax.axhline(5,color='gray',ls='--');ax.axhline(40,color='gray',ls=':')
        ax.set(xlabel='扩展范围后的追加步骤',ylabel='泵浦 / mW',title='扩展试验中的实际泵浦轨迹')
    extra_curve=image('扩展范围增广残差',expanded_curve);pump_fig=image('扩展范围泵浦轨迹',pump_curve)
    expanded_state=np.load(ROOT/'bordered_expanded.npz')
    def expanded_output(ax):
        p=np.sum(abs(expanded_state['output'])**2,axis=0);t=(np.arange(len(p))-len(p)/2)*.125
        ax.plot(t,p);ax.set(xlabel='时间 / ps',ylabel='输出瞬时功率 / W',title='扩展范围试验停止态输出')
    expanded_out=image('扩展范围停止态输出',expanded_output)
    extra=(f"<section><h2>6．5–60 mW范围的独立扩展试验</h2><p>局部试验碰到40 mW人为边界后，保留其全部结果；另将q下限扩到−9，对应5 mW。初态沿用局部末态，原截面、场尺度和规范不变；信赖半径重置为0.1，最多25步。它不能被称为50 mW附近的小范围求根。</p>"
        f"<p class='note'>停止：{expanded_status}。泵浦{1000*expanded['value']:.6f} mW；原方程残差{expanded['physical_residual']:.7g}，截面残差{expanded['hyperplane_residual']:.3g}；输出{expanded['output_energy_nJ']:.6f} nJ。实际追加{eh[-1]['step']}步，{expanded['seconds']:.1f}秒、{expanded['evaluations']}次映射。</p>"
        +linear_note+extra_curve+pump_fig+expanded_out+"<p>是否得到根以原方程残差和非零局域性为准，不能以泵浦下降或能量接近目标代替验收。尚未认证的结果不用于正式伪弧长或Floquet。</p></section>")
    body=body.replace('<section><h2>6．对分支假说的当前判断</h2>',extra+'<section><h2>7．对分支假说的当前判断</h2>')
    body=body.replace('<h2>7．复现与验收记录</h2>','<h2>8．复现与验收记录</h2>')
    if (ROOT/'expanded_endpoint_check.json').exists():
        check=read('expanded_endpoint_check.json')
        body=body.replace('<h2>6．5–60 mW范围的独立扩展试验</h2>', '<h2>6．5–60 mW范围的独立扩展试验</h2>'
            f"<p>末态相对场闭合残差{check['relative_field_residual']:.6g}，腔内采样点场能量{check['field_energy_nJ']:.6g} nJ，最大反转差{check['population_gap']:.5g}。这些量用于区分闭合改善与场幅度下降；不能仅凭固定尺度绝对残差下降宣称非零根已收敛。4.096 ns复核的完整残差向量相对差{100*check['full_vector_relative_change']:.4f}%。</p>")
for filename,label in [('parameter_geometry.json','下载完整参数与梯度诊断'),('bordered_expanded.json','下载扩展范围求解记录')]:
    if (ROOT/filename).exists():
        url='data:application/json;base64,'+base64.b64encode((ROOT/filename).read_bytes()).decode()
        body=body.replace('</main>',f'<p><a download="{filename}" href="{url}">{label}</a></p></main>')
path=ROOT/'原生10MHz_参数耦合与增广求根验证.html';path.write_text(body,encoding='utf8');print(path)
