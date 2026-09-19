"""Separate 5–60 mW range test; preserve original anchor, normal and state units."""
import json,time,hashlib
import numpy as np
from scipy.signal import find_peaks
from config import ROOT,PROJECT
from steady_bordered import BorderedResidual
from steady_window import build_localized
from steady_hookstep import solve_globalized
from adaptive_solver import health

prior=json.loads((ROOT/'bordered_pilot.json').read_text())
assert prior['name']=='pump' and prior['q']< -1.9
state=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
local=np.load(ROOT/'bordered_pilot.npz')
residual=BorderedResidual(state,'pump',local['normal'],gpu=True,q_bounds=(-9.,2.))
trace=[]
def progress(row):print(json.dumps(row),flush=True)
def observer(step,y,r,d):
    trace.append(dict(step=step,pump_W=float(.05+.005*y[-1]),q=float(y[-1])))
    (ROOT/'expanded_parameter_trace.json').write_text(json.dumps(trace,indent=2),encoding='utf8')
    print('PUMP',json.dumps(trace[-1]),flush=True)
hashes={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in
        ['steady_parameters.py','steady_bordered.py','steady_hookstep.py','steady_diagnostics.py','extend_bordered_pump.py']}
start=time.perf_counter()
y,history,status=solve_globalized(residual,local['y'],max_steps=25,radius=.1,stop_window=5,
    preconditioner_builder=build_localized,progress=progress,observer=observer)
r=residual(y);a,p,phase,shift=residual.unpack(y);power=np.sum(abs(residual.output)**2,axis=0)
trace.append(dict(step=history[-1]['step'],pump_W=float(.05+.005*y[-1]),q=float(y[-1])))
result=dict(name='pump',value=float(.05+.005*y[-1]),unit='W',q=float(y[-1]),status=status,
    seconds=time.perf_counter()-start,evaluations=residual.evaluations,
    augmented_residual=float(np.linalg.norm(r)),physical_residual=float(np.linalg.norm(r[:-1])),
    hyperplane_residual=float(abs(r[-1])),output_energy_nJ=float(power.sum()*.125/1000),
    peaks=int(len(find_peaks(power,prominence=.1*power.max())[0])),time_edge=health(a[None],.125)[0],
    spectral_edge=health(a[None],.125)[1],history=history,parameter_trace=trace,
    code_sha256=hashes,certified_stable=False,bounds_scaled=[-9,2],initial_radius=.1,
    scope='separate expanded-range test, original hyperplane preserved; not evidence of a root near 50 mW')
(ROOT/'bordered_expanded.json').write_text(json.dumps(result,indent=2),encoding='utf8')
np.savez_compressed(ROOT/'bordered_expanded.npz',y=y,x=y[:-1],a=a,pop=p,output=residual.output,
    normal=local['normal'],original_template=state['original_template'],gauge_template=residual.template,
    time_tangent=residual.time_tangent,scale=residual.scale,time_scale=residual.time_scale,phase=phase,shift=shift,dt=.125)
print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ('history','parameter_trace')}),flush=True)
