"""Nonlinear solve after same-state broad-band preconditioner acceptance."""
import os,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from config import ROOT,PROJECT,config
from steady_state import CavityResidual,solve
from adaptive_solver import health

saved=np.load(Path(os.environ['LASER_STEADY_DIR'])/'best_pilot_larger_krylov.npz')
residual=CavityResidual(config('CNT_OC',.2),.125,saved['a'],.05,.8,gpu=True)
x=np.load(ROOT/'linear_none.npz')['x']
def progress(row):
    print(json.dumps(row),flush=True)
    (ROOT/'newton_progress.json').write_text(json.dumps(row),encoding='utf8')
start=time.perf_counter()
x,history,status=solve(residual,x,max_steps=12,inner=60,precondition='coarse',
                        coarse_cutoff=384,rebuild_every=3,progress=progress)
r=residual(x);a,p,phase,shift=residual.unpack(x)
power=np.sum(abs(residual.output)**2,axis=0)
edges=health(a[None],.125)
row=dict(status=status,seconds=time.perf_counter()-start,evaluations=residual.evaluations,
    residual=float(np.linalg.norm(r)),relative_field_residual=float(np.linalg.norm(r[:4*residual.n])*residual.scale/np.linalg.norm(a)),
    population_gap=float(np.max(abs(p-residual.neq))),output_energy_nJ=float(power.sum()*.125/1000),
    peaks=int(len(find_peaks(power,prominence=.1*power.max())[0])),time_edge=edges[0],spectral_edge=edges[1],
    history=history,certified_stable=False,
    settings=dict(cutoff=384,rebuild_every=3,max_steps=12,inner=60),
    code_sha256={f:hashlib.sha256((PROJECT/f).read_bytes()).hexdigest() for f in
                 ['steady_state.py','steady_preconditioner.py','run_wide_newton.py']})
np.savez_compressed(ROOT/'wide_newton.npz',a=a,pop=p,x=x,template=saved['a'],dt=.125,phase=phase,
                    shift=shift,output=residual.output)
(ROOT/'wide_newton.json').write_text(json.dumps(row,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in row.items() if k not in ['history','code_sha256']}),flush=True)
