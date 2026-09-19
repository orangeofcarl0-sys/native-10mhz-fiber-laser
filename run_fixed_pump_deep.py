"""Warm-start inner solves; output energy never enters the residual."""
import json,time,hashlib
import numpy as np
from scipy.signal import find_peaks
from config import ROOT,PROJECT
from steady_parameters import parameter_residual
from steady_support_lu import build_factored
from steady_hookstep import solve_globalized,arnoldi
from adaptive_solver import health
state=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
prior=PROJECT/'results/steady_energy_20260919'
start_state=np.load(prior/'fixed_final.npz')
pump0=json.loads((prior/'fixed_final.json').read_text())['pump_W']
benchmark=json.loads((ROOT/'support_benchmark.json').read_text())
options=[r for r in benchmark['rows'] if r['budget']==240]
best=min(options,key=lambda r:max(r['true_relative'],r['central_relative']))
label=best['support']
protocol=dict(support=label,linear_limit=240,linear_tolerance=.008,max_steps=30,rebuild_every=30,linear_refresh_threshold=.01,
    stop_window=None,initial_radius=.05,root_tolerance=1e-7,
    jv_method='central',jv_epsilon=1e-5,jv_check_epsilon=1e-6,
    selection='smallest verified 240-vector linear residual at benchmark state; not globally optimal',
    benchmark_below_one_percent=bool(max(best['true_relative'],best['central_relative'])<.01))
(ROOT/'deep_protocol.json').write_text(json.dumps(protocol,indent=2))
probe_residual=parameter_residual(state,'pump',(pump0-.05)/.005,gpu=True)
probe_x=start_state['x'];probe_r=probe_residual(probe_x)
start=time.perf_counter();probe_pre,probe_info=build_factored(probe_residual,probe_x,probe_r,label)
factor_seconds=time.perf_counter()-start
def jv(v):
    eps=1e-7/max(np.linalg.norm(v),1e-100)
    return (probe_residual(probe_x+eps*v)-probe_r)/eps
hh,zz,yy,_=arnoldi(jv,probe_r,probe_pre,limit=240,tolerance=0.)
d=zz@yy;eta=float(np.linalg.norm(probe_r+jv(d))/np.linalg.norm(probe_r))
eps=1e-6/max(np.linalg.norm(d),1e-100)
central=(probe_residual(probe_x+eps*d)-probe_residual(probe_x-eps*d))/(2*eps)
eta_central=float(np.linalg.norm(probe_r+central)/np.linalg.norm(probe_r))
validation=dict(support=label,factor_seconds=factor_seconds,evaluations=probe_residual.evaluations,LU_linear_relative=eta,
    SVD_linear_relative=best['true_relative'],LU_central_relative=eta_central,SVD_central_relative=best['central_relative'],**probe_info)
(ROOT/'factor_validation.json').write_text(json.dumps(validation,indent=2))
assert abs(eta_central-best['central_relative'])<max(5e-4,.05*best['central_relative']),validation
print('FACTORIZATION',json.dumps(validation),flush=True)
cache=[(probe_pre,probe_info)]
def builder(residual,x,r,cutoff=384):
    if cache and np.array_equal(x,probe_x):return cache.pop()
    return build_factored(residual,x,r,label)
sources=['steady_hookstep.py','steady_support.py','steady_support_lu.py','steady_state.py','steady_parameters.py','run_fixed_pump_deep.py']
hashes={p:hashlib.sha256((PROJECT/p).read_bytes()).hexdigest() for p in sources}
finals={};summary=[]
for name,pump,parent in [('anchor',pump0,None),('down_275',.0275,'anchor'),('down_2725',.02725,'down_275'),
                        ('down_270',.027,'down_2725'),('up_2775',.02775,'anchor'),('up_280',.028,'up_2775')]:
    initial=start_state['x'] if parent is None else finals[parent]
    residual=parameter_residual(state,'pump',(pump-.05)/.005,gpu=True)
    started=time.perf_counter();trace=[]
    def progress(row):
        print(name,json.dumps(row),flush=True)
        (ROOT/(name+'_progress.json')).write_text(json.dumps(row,indent=2))
    def observer(step,x,r,d):
        trace.append(dict(step=step,physical_residual=float(np.linalg.norm(r))))
        np.savez_compressed(ROOT/(name+'_checkpoint.npz'),x=x)
    x,h,status=solve_globalized(residual,initial,max_steps=30,radius=.05,rebuild_every=30,
        linear_limit=240,linear_tolerance=.008,linear_refresh_threshold=.01,stop_window=None,
        preconditioner_builder=builder,progress=progress,observer=observer,
        jv_method='central',jv_epsilon=1e-5,jv_check_epsilon=1e-6)
    r=residual(x);a,p,phase,shift=residual.unpack(x);out=residual.output
    energy=float(np.sum(abs(out)**2)*residual.dt/1000)
    lines=[v['true_newton_linear_residual'] for v in h if 'true_newton_linear_residual' in v]
    result=dict(name=name,pump_W=pump,parent=parent,status=status,physical_residual=float(np.linalg.norm(r)),
        output_energy_nJ=energy,relative_field_residual=float(np.linalg.norm(r[:4*residual.n])*residual.scale/np.linalg.norm(a)),
        population_gap=float(np.max(abs(p-residual.neq))),peaks=int(len(find_peaks(np.sum(abs(out)**2,axis=0),prominence=.1*np.max(np.sum(abs(out)**2,axis=0)))[0])),
        time_edge=health(a[None],residual.dt)[0],spectral_edge=health(a[None],residual.dt)[1],
        linear_fraction_below_one_percent=float(np.mean(np.array(lines)<.01)),
        seconds=time.perf_counter()-started,evaluations=residual.evaluations,history=h,protocol=protocol,
        code_sha256=hashes,numerical_root=bool(np.linalg.norm(r)<1e-7),certified_stable=False)
    (ROOT/(name+'.json')).write_text(json.dumps(result,indent=2))
    np.savez_compressed(ROOT/(name+'.npz'),x=x,a=a,pop=p,output=out,original_template=state['original_template'],
        gauge_template=residual.template,time_tangent=residual.time_tangent,scale=residual.scale,time_scale=residual.time_scale,dt=residual.dt)
    finals[name]=x;summary.append({k:v for k,v in result.items() if k not in ('history','code_sha256')})
    (ROOT/'deep_summary.json').write_text(json.dumps(summary,indent=2))
    print('RESULT',json.dumps(summary[-1]),flush=True)
