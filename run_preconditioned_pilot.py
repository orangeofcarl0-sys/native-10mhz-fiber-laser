"""Paired control: identical state, template, scaling, grid and Newton budgets."""
import os, json, time, hashlib
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from config import ROOT, PROJECT, config
from steady_state import CavityResidual, solve
from adaptive_solver import health


def run():
    source = Path(os.environ['LASER_STEADY_DIR'])/'best_pilot_larger_krylov.npz'
    saved = np.load(source)
    a, p = saved['a'], saved['pop']
    # Both controls use the same template and identical one-map phase estimate.
    # No comparison against previous differently scaled histories is claimed.
    c = config('CNT_OC', .2)
    rows = []
    for enabled in [False, True]:
        residual = CavityResidual(c,.125,a,.05,.8,gpu=True)
        x0 = residual.pack(a,p)
        initial = residual(x0)
        m = 2*a.shape[-1]
        mapped = a/residual.scale+(initial[:m]+1j*initial[m:2*m]).reshape(a.shape)
        x0[-2] = np.angle(np.vdot(a,mapped))
        start = time.perf_counter()
        x,history,status = solve(residual,x0,max_steps=10,inner=60,precondition=enabled)
        r = residual(x)
        field,pop,phase,shift = residual.unpack(x)
        power = np.sum(abs(residual.output)**2,axis=0)
        edges = health(field[None],.125)
        row = dict(precondition=enabled,status=status,seconds=time.perf_counter()-start,
                   evaluations=residual.evaluations,residual=float(np.linalg.norm(r)),
                   relative_field_residual=float(np.linalg.norm(r[:4*len(power)])*residual.scale/np.linalg.norm(field)),
                   population_gap=float(np.max(abs(pop-residual.neq))),
                   output_energy_nJ=float(power.sum()*.125/1000),
                   peaks=int(len(find_peaks(power,prominence=.1*power.max())[0])),
                   time_edge=edges[0],spectral_edge=edges[1],history=history,certified_stable=False)
        rows.append(row)
        np.savez_compressed(ROOT/f'block_{enabled}.npz',a=field,pop=pop,x=x,dt=.125,
                            template=a,phase=phase,shift=shift,output=residual.output)
        payload = dict(input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            settings=dict(topology='CNT_OC',oc=.8,pump_W=.05,gdd_ps2=.2,n=8192,dt_ps=.125,
                          max_steps=10,inner=60),
            code_sha256={f:hashlib.sha256((PROJECT/f).read_bytes()).hexdigest() for f in
                         ['steady_state.py','steady_preconditioner.py','run_preconditioned_pilot.py']},trials=rows)
        (ROOT/'block_pilot.json').write_text(json.dumps(payload,indent=2),encoding='utf8')
        print(json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)


if __name__ == '__main__':
    run()
