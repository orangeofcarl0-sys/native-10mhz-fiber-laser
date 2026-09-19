"""40-step paired experiment at fixed pump; no energy constraint or continuation."""
import json,hashlib,time
import numpy as np
from config import ROOT,PROJECT
from steady_parameters import parameter_residual
from steady_support_lu import build_factored
from steady_augmented_outer import solve_augmented

pump=float(__import__('os').environ.get('LASER_TEST_PUMP','0.0275'))
name='down_275' if pump==.0275 else 'down_270'
input_path=PROJECT/'results/steady_fixed_pump_20260919'/f'{name}.npz'
initial=np.load(input_path)['x']
template=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
sources=['run_augmented_pair.py','steady_augmented_outer.py','steady_cauchy.py','steady_hookstep.py','steady_support_lu.py','steady_state.py','steady_parameters.py']
hashes={p:hashlib.sha256((PROJECT/p).read_bytes()).hexdigest() for p in sources}
protocol=dict(pump_W=pump,max_steps=40,initial_radius=.003125,radius_cap=2.,linear_limit=240,
    linear_tolerance=.008,independent_linear_gate=.01,jv_epsilon=1e-5,check_epsilon=1e-6,
    coarse_difference='forward',coarse_epsilon=1e-6,support='C',root_tolerance=1e-7,
    refresh_events=['initial','linear_gate','low_rho<0.25','model_discrepancy>0.05','B: non_descent'],
    input=str(input_path.relative_to(PROJECT)).replace('\\','/'),input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest())
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
for arm in ['A','B']:
    f=parameter_residual(template,'pump',(pump-.05)/.005,gpu=True);start=time.perf_counter()
    def builder(f,x,r,cache):
        print('BUILD',arm,float(np.linalg.norm(r)),flush=True)
        return build_factored(f,x,r,'C',descent=cache)
    def observe(x,r,h):
        row=h[-1];t=row['trials'][row['accepted_trial']] if 'accepted_trial' in row else {}
        print('STEP',arm,row['step'],'R',float(np.linalg.norm(r)),'rho',t.get('rho'),'radius',row.get('next_radius'),'kind',t.get('kind'),flush=True)
        np.savez_compressed(ROOT/(arm+'_checkpoint.npz'),x=x)
        (ROOT/(arm+'_progress.json')).write_text(json.dumps(dict(history=h,residual=float(np.linalg.norm(r)),seconds=time.perf_counter()-start),indent=2))
    x,h,status=solve_augmented(f,initial,augmented=arm=='B',builder=builder,observer=observe)
    r=f(x);a,p,phase,shift=f.unpack(x)
    result=dict(arm=arm,status=status,physical_residual=float(np.linalg.norm(r)),initial_residual=h[0]['residual'],
        accepted_steps=sum('accepted_trial' in v for v in h),outer_attempts=len(h),seconds=time.perf_counter()-start,
        evaluations=f.evaluations,history=h,protocol=protocol,code_sha256=hashes,
        numerical_root=bool(np.linalg.norm(r)<1e-7),certified_stable=False,
        output_energy_nJ=float(np.sum(abs(f.output)**2)*f.dt/1000))
    np.savez_compressed(ROOT/(arm+'_final.npz'),x=x,output=f.output)
    (ROOT/(arm+'.json')).write_text(json.dumps(result,indent=2))
    print('RESULT',arm,status,result['physical_residual'],result['accepted_steps'],flush=True)
