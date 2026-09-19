"""Discriminate Krylov budget and initial-state limitations, not physical time."""
import os, json, time, hashlib
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks
from config import ROOT, PROJECT, config
from steady_state import CavityResidual, solve
from adaptive_solver import health

c = config('CNT_OC', .2)
previous = np.load(ROOT/'steady_trial_2.npz')
source = loadmat(os.path.join(os.environ['LASER_SCAN_DIR'],'fine_group_08.mat'),
                 variable_names=['a','pop'])
rows = []
for name,a,p,phase,shift in [
    ('best_pilot_larger_krylov',previous['a'],previous['pop'],1.108546071929818,-.030333283383558656),
    ('saved_RT600',source['a'][53],source['pop'][53],0.,0.)]:
    residual = CavityResidual(c,.125,a,.05,.8,gpu=True)
    # Reanchor only the gauges; retain the actual field and inversion.
    x = residual.pack(a,p,phase,shift)
    start = time.perf_counter()
    x,history,status = solve(residual,x,max_steps=20,inner=80)
    r = residual(x)
    a,p,phase,shift = residual.unpack(x)
    power = np.sum(abs(residual.output)**2,axis=0)
    edges = health(a[None],.125)
    row = dict(name=name,status=status,residual=float(np.linalg.norm(r)),
        relative_field_residual=float(np.linalg.norm(r[:4*8192])*residual.scale/np.linalg.norm(a)),
        population_gap=float(np.max(abs(p-residual.neq))),
        output_energy_nJ=float(power.sum()*.125/1000),
        peaks=int(len(find_peaks(power,prominence=.1*power.max())[0])),
        time_edge=edges[0],spectral_edge=edges[1],seconds=time.perf_counter()-start,
        evaluations=residual.evaluations,history=history,certified_stable=False)
    rows.append(row)
    np.savez_compressed(ROOT/f'{name}.npz',a=a,pop=p,x=x,dt=.125)
    (ROOT/'steady_followup.json').write_text(json.dumps(dict(
        source_sha256={f:hashlib.sha256((PROJECT/f).read_bytes()).hexdigest() for f in
                       ['steady_state.py','spectral_engine.py','continue_steady_pilot.py']},
        trials=rows),indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)
