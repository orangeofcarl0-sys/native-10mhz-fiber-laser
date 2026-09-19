"""Budgeted fixed-pump controls, then a 0.9 nJ physical closure trial."""
import json,time,hashlib
import numpy as np
from config import ROOT,PROJECT
from steady_parameters import parameter_residual
from steady_energy import EnergyResidual
from steady_leakage import build_leakage
from steady_window import build_localized
from steady_hookstep import solve_globalized
from adaptive_solver import health
from scipy.signal import find_peaks
state=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
prior=PROJECT/'results/steady_parameter_20260919'
last=np.load(prior/'bordered_expanded.npz');local=np.load(prior/'bordered_pilot.npz')
control=json.loads((ROOT/'leakage_fixed_control.json').read_text())['rows']
# Require actual improvement before changing the production preconditioner.
base=control[0]
best=min(control,key=lambda v:v['true_residual'])
selected=best if best['true_residual']<.95*base['true_residual'] else base
mode=selected['mode'];threshold=selected['threshold']
def builder(r,x,fx,cutoff=384):
    if mode=='baseline':return build_localized(r,x,fx,cutoff)
    return build_leakage(r,x,fx,cutoff,mode,threshold)
hashes={p:hashlib.sha256((PROJECT/p).read_bytes()).hexdigest() for p in
    ['steady_state.py','steady_energy.py','steady_leakage.py','steady_hookstep.py','run_energy_profiles.py']}
for label,pump,initial in [('fixed_40',.04,local['x']),('fixed_35',.035,last['x']),
                          ('fixed_30',.03,last['x']),('fixed_final',.05+.005*last['y'][-1],last['x']),
                          ('energy_09',None,last['y'])]:
    augmented=pump is None
    residual=EnergyResidual(state,.9,gpu=True) if augmented else parameter_residual(state,'pump',(pump-.05)/.005,gpu=True)
    start=time.perf_counter();trace=[]
    def progress(row):print(label,json.dumps(row),flush=True)
    def observer(step,x,r,d):
        trace.append(dict(step=step,physical_residual=float(np.linalg.norm(r[:-1] if augmented else r)),
            energy_equation=float(r[-1]) if augmented else None))
        np.savez_compressed(ROOT/(label+'_checkpoint.npz'),x=x)
    x,h,status=solve_globalized(residual,initial,max_steps=20 if augmented else 8,radius=.1,
        stop_window=5,preconditioner_builder=builder,progress=progress,observer=observer)
    r=residual(x);a,p,phase,shift=residual.unpack(x);out=residual.output
    energy=float(np.sum(abs(out)**2)*residual.dt/1000)
    actual_pump=float(.05+.005*x[-1]) if augmented else float(pump)
    result=dict(label=label,status=status,pump_W=actual_pump,physical_residual=float(np.linalg.norm(r[:-1] if augmented else r)),
        augmented_residual=float(np.linalg.norm(r)),energy_relative_error=float(r[-1]) if augmented else None,
        output_energy_nJ=energy,relative_field_residual=float(np.linalg.norm(r[:4*residual.n])*residual.scale/np.linalg.norm(a)),
        peaks=int(len(find_peaks(np.sum(abs(out)**2,axis=0),prominence=.1*np.max(np.sum(abs(out)**2,axis=0)))[0])),
        time_edge=health(a[None],residual.dt)[0],spectral_edge=health(a[None],residual.dt)[1],
        seconds=time.perf_counter()-start,evaluations=residual.evaluations,history=h,trace=trace,
        selected_preconditioner=dict(mode=mode,threshold=threshold),code_sha256=hashes,certified_stable=False,
        initial_source='saved local40mW' if label=='fixed_40' else 'same saved27.744mW endpoint',
        initial_pump_W=float(.05+.005*(local['y'][-1] if label=='fixed_40' else last['y'][-1])))
    (ROOT/(label+'.json')).write_text(json.dumps(result,indent=2))
    np.savez_compressed(ROOT/(label+'.npz'),x=x,a=a,pop=p,output=out,original_template=state['original_template'],
        gauge_template=residual.template,time_tangent=residual.time_tangent,scale=residual.scale,time_scale=residual.time_scale,dt=residual.dt)
    print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ('history','trace','code_sha256')}),flush=True)
