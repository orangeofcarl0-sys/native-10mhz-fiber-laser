"""Same Arnoldi basis builder and preconditioner; only globalization differs."""
import json,time,hashlib
import numpy as np
from scipy.signal import find_peaks
from config import ROOT,PROJECT,config
from steady_state import CavityResidual
from steady_hookstep import solve_globalized
from steady_diagnostics import gauge_geometry
from adaptive_solver import health

diagnostics=json.loads((ROOT/'taylor_diagnostics.json').read_text())
assert len(diagnostics)==4
assert max(row['epsilon_sweep'][2]['forward_relative'] for row in diagnostics)<1e-3
source=PROJECT/'results/steady_block_20260919'
initial=np.load(source/'input/best_pilot_larger_krylov.npz')
x0=np.load(source/'spectral_support.npz')['x']
hashes={f:hashlib.sha256((PROJECT/f).read_bytes()).hexdigest() for f in
        ['steady_state.py','steady_preconditioner.py','steady_hookstep.py','steady_diagnostics.py','run_hookstep_pair.py']}
rows=[]
conditions=[('line_search',None),('hookstep',None)]
for name,threshold in conditions:
    residual=CavityResidual(config('CNT_OC',.2),.125,initial['a'],.05,.8,gpu=True)
    def progress(row):
        print(json.dumps(dict(name=name,**row)),flush=True)
        (ROOT/'hookstep_progress.json').write_text(json.dumps(dict(name=name,**row)),encoding='utf8')
    start=time.perf_counter()
    x,history,status=solve_globalized(residual,x0,mode='line_search' if name=='line_search' else 'hookstep',
        max_steps=12,reanchor_threshold=threshold,progress=progress)
    r=residual(x);a,p,phase,shift=residual.unpack(x)
    power=np.sum(abs(residual.output)**2,axis=0);edges=health(a[None],.125)
    row=dict(name=name,status=status,seconds=time.perf_counter()-start,evaluations=residual.evaluations,
        residual=float(np.linalg.norm(r)),relative_field_residual=float(np.linalg.norm(r[:4*residual.n])*residual.scale/np.linalg.norm(a)),
        population_gap=float(np.max(abs(p-residual.neq))),output_energy_nJ=float(power.sum()*.125/1000),
        peaks=int(len(find_peaks(power,prominence=.1*power.max())[0])),time_edge=edges[0],spectral_edge=edges[1],
        gauge=gauge_geometry(residual,x),history=history,certified_stable=False)
    rows.append(row)
    np.savez_compressed(ROOT/f'{name}.npz',x=x,a=a,pop=p,output=residual.output,
        original_template=initial['a'],gauge_template=residual.template,time_tangent=residual.time_tangent,
        scale=residual.scale,time_scale=residual.time_scale,phase=phase,shift=shift,dt=.125)
    payload=dict(code_sha256=hashes,settings=dict(max_steps=12,krylov_limit=120,cutoff=384,
        initial_radius=.1,rebuild_every=3,step_metric='identity in fixed scaled state coordinates'),trials=rows)
    (ROOT/'hookstep_pair.json').write_text(json.dumps(payload,indent=2),encoding='utf8')
    print('RESULT',json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)
    if name=='hookstep' and max(h['gauge']['condition'] for h in history)>20:
        conditions.append(('hookstep_reanchor',20.))
