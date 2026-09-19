"""Select an informative local knob, then solve the full bordered equations."""
import json,time,hashlib
import numpy as np
from scipy.signal import find_peaks
from config import ROOT,PROJECT
from steady_parameters import PARAMETERS
from steady_bordered import BorderedResidual
from steady_preconditioner import spectral_coordinates
from steady_window import embed,build_localized
from steady_hookstep import solve_globalized
from adaptive_solver import health

diagnosis=json.loads((ROOT/'parameter_geometry.json').read_text())
candidates=[p for p in diagnosis['parameters'] if p['halving_relative_difference']<1e-3
            and p['eta_projected']>1e-3]
if not candidates:raise RuntimeError('No parameter passes the local derivative/coupling/range screen')
selected=max(candidates,key=lambda p:p['eta_projected']);name=selected['name']
state=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
n=state['a'].shape[-1]//2
_,lift,_=spectral_coordinates(n,len(state['pop']),384)
modes=np.load(ROOT/'parameter_geometry_modes.npz');normal=embed(lift(modes['right'][-1]),n)
residual=BorderedResidual(state,name,normal,gpu=True);y=np.r_[state['x'],0.]
hashes={f:hashlib.sha256((PROJECT/f).read_bytes()).hexdigest() for f in
        ['steady_parameters.py','steady_bordered.py','steady_hookstep.py','steady_diagnostics.py','run_bordered_pilot.py']}
def progress(row):
    print(json.dumps(row),flush=True)
def observer(step,y,r,d):
    base,scale,_=PARAMETERS[name]
    print('PARAM',json.dumps(dict(step=step,name=name,value=base+scale*y[-1],q=float(y[-1]))),flush=True)
    np.savez_compressed(ROOT/'bordered_last_direction.npz',step=step,y=y,r=r,d=d)
start=time.perf_counter()
y,history,status=solve_globalized(residual,y,max_steps=20,radius=.05,stop_window=5,
    preconditioner_builder=build_localized,progress=progress,observer=observer)
r=residual(y);a,p,phase,shift=residual.unpack(y);power=np.sum(abs(residual.output)**2,axis=0)
base,scale,unit=PARAMETERS[name]
result=dict(name=name,value=float(base+scale*y[-1]),unit=unit,q=float(y[-1]),status=status,
    seconds=time.perf_counter()-start,evaluations=residual.evaluations,
    augmented_residual=float(np.linalg.norm(r)),physical_residual=float(np.linalg.norm(r[:-1])),
    hyperplane_residual=float(abs(r[-1])),output_energy_nJ=float(power.sum()*.125/1000),
    peaks=int(len(find_peaks(power,prominence=.1*power.max())[0])),time_edge=health(a[None],.125)[0],
    spectral_edge=health(a[None],.125)[1],history=history,code_sha256=hashes,selected=selected,
    certified_stable=False,bounds_scaled=[-2,2],selection='largest normalized weakest-left coupling among reliable derivatives; full linear parameter step exceeds local bounds, so trust-region trial retains explicit bounds')
(ROOT/'bordered_pilot.json').write_text(json.dumps(result,indent=2),encoding='utf8')
np.savez_compressed(ROOT/'bordered_pilot.npz',y=y,x=y[:-1],a=a,pop=p,output=residual.output,
    normal=normal,original_template=state['original_template'],gauge_template=residual.template,
    time_tangent=residual.time_tangent,scale=residual.scale,time_scale=residual.time_scale,phase=phase,shift=shift,dt=.125)
print('RESULT',json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
