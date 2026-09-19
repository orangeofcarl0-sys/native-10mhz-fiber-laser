"""Same-state shifted-direction pilot after a progress stop, not a PTC root claim."""
import json,time
import numpy as np
from config import ROOT,config
from steady_window import restore
from steady_ptc import mass_sign,local_matrix,shifted_local_inverse
from steady_hookstep import arnoldi
from steady_diagnostics import blocks

previous=json.loads((ROOT/'tail_continuation.json').read_text())
if previous['status'] not in ('progress_stagnated','globalization_stalled','radius_floor_reached'):
    raise RuntimeError('Do not switch methods solely because the step budget expired')
state=np.load(ROOT/'tail_continuation.npz');residual,x=restore(config('CNT_OC',.2),state,gpu=True)
r=residual(x);beta=np.linalg.norm(r);sign=mass_sign(residual.n,residual.cells)
start=time.perf_counter();matrix=local_matrix(residual,x,r);rows=[]
def jv(v):
    epsilon=1e-7/max(np.linalg.norm(v),1e-100)
    return (residual(x+epsilon*v)-r)/epsilon
for mu in [0.,.001,.01,.1,1.]:
    inverse=shifted_local_inverse(matrix,residual.n//2,residual.cells,mu)
    h,z,y,relative=arnoldi(lambda v:jv(v)+mu*sign*v,r,inverse)
    direction=z@y;response=jv(direction)
    row=dict(mu=mu,arnoldi_relative=relative,
        true_shifted_relative=float(np.linalg.norm(r+response+mu*sign*direction)/beta),
        unshifted_relative=float(np.linalg.norm(r+response)/beta),
        direction_blocks=blocks(residual,direction),trials=[])
    # Equal actual-step caps avoid comparing enormous Newton and tiny PTC steps.
    for cap in [.0125,.025,.05,.1]:
        alpha=min(1.,cap/max(np.linalg.norm(direction),1e-100));step=alpha*direction
        trial=dict(cap=cap,alpha=alpha,feasible=bool(residual.feasible(x+step)))
        if trial['feasible']:
            rr=residual(x+step);prediction=.5*(beta**2-np.linalg.norm(r+alpha*response)**2)
            actual=.5*(beta**2-np.linalg.norm(rr)**2)
            trial.update(residual=float(np.linalg.norm(rr)),actual_reduction=float(actual),
                         unshifted_prediction=float(prediction),rho=float(actual/prediction) if prediction>0 else None)
        row['trials'].append(trial)
    rows.append(row);print('SHIFT',json.dumps(row),flush=True)
    (ROOT/'ptc_direction_pilot.json').write_text(json.dumps(dict(initial_residual=float(beta),rows=rows,
        seconds=time.perf_counter()-start,evaluations=residual.evaluations,
        scope='same-state direction pilot; no multi-step PTC trajectory or physical stability claim'),indent=2),encoding='utf8')
