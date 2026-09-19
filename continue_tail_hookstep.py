"""Wide-window continuation with unchanged physical state and a progress stop."""
import json,time,hashlib
import numpy as np
from scipy.signal import find_peaks
from config import ROOT,PROJECT,config
from steady_window import restore,build_localized
from steady_hookstep import solve_globalized
from adaptive_solver import health

window=json.loads((ROOT/'endpoint_window.json').read_text())
assert window['full_vector_relative_change']>1e-3
state=np.load(PROJECT/'results/steady_hookstep_20260919/hookstep.npz')
residual,x=restore(config('CNT_OC',.2),state,wide=True,gpu=True)
hashes={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in
        ['steady_hookstep.py','steady_window.py','steady_state.py','steady_preconditioner.py','continue_tail_hookstep.py']}
def progress(row):
    print(json.dumps(row),flush=True)
    (ROOT/'continuation_progress.json').write_text(json.dumps(row),encoding='utf8')
def observer(step,x,r,d):
    np.savez_compressed(ROOT/'continuation_last_direction.npz',step=step,x=x,r=r,d=d)
start=time.perf_counter()
x,history,status=solve_globalized(residual,x,max_steps=25,radius=.0125,stop_window=5,
    preconditioner_builder=build_localized,observer=observer,progress=progress)
r=residual(x);a,p,phase,shift=residual.unpack(x)
power=np.sum(abs(residual.output)**2,axis=0)
result=dict(status=status,seconds=time.perf_counter()-start,evaluations=residual.evaluations,
    residual=float(np.linalg.norm(r)),relative_field_residual=float(np.linalg.norm(r[:4*residual.n])*residual.scale/np.linalg.norm(a)),
    population_gap=float(np.max(abs(p-residual.neq))),output_energy_nJ=float(power.sum()*.125/1000),
    peaks=int(len(find_peaks(power,prominence=.1*power.max())[0])),
    time_edge=health(a[None],.125)[0],spectral_edge=health(a[None],.125)[1],history=history,
    code_sha256=hashes,certified_stable=False,
    settings=dict(window_ps=2048,dt_ps=.125,max_steps=25,initial_radius=.0125,
        preconditioner='3093-dimensional centered localized Fourier subspace at ±375 GHz; outer field -I',
        stop='last 5 accepted steps decrease <2% AND at least 4 rho>0.5'))
(ROOT/'tail_continuation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
np.savez_compressed(ROOT/'tail_continuation.npz',x=x,a=a,pop=p,output=residual.output,
    original_template=np.pad(state['original_template'],((0,0),(4096,4096))),
    gauge_template=residual.template,time_tangent=residual.time_tangent,
    scale=residual.scale,time_scale=residual.time_scale,phase=phase,shift=shift,dt=.125)
print('RESULT',json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
