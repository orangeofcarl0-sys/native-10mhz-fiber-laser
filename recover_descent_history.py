"""Recover omitted fresh directions by exact replay, without a new search."""
import json,time,hashlib
import numpy as np
from config import ROOT,PROJECT
from steady_parameters import parameter_residual
from steady_support_lu import build_factored
from steady_augmented_outer import solve_augmented
old=json.loads((PROJECT/'results/steady_step_gate_20260920/continuation.json').read_text())
x0=np.load(PROJECT/'results/steady_augmented_pair_20260920/B_final.npz')['x']
template=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
f=parameter_residual(template,'pump',(.0275-.05)/.005,gpu=True)
directions=[];states=[];start=time.perf_counter()
def builder(f,x,r,cache):
    print('RECOVER BUILD',len(directions),float(np.linalg.norm(r)),flush=True)
    pre,info=build_factored(f,x,r,'C',descent=cache)
    d=-cache['lift'](cache['gradient']);d/=np.linalg.norm(d)
    directions.append(d);states.append(x.copy())
    np.savez_compressed(ROOT/'history_directions.npz',directions=np.array(directions),states=np.array(states))
    return pre,info
def observer(x,r,h):
    row=h[-1];prior=old['history'][len(h)-1];t=row['trials'][row['accepted_trial']]
    assert abs(t['residual']-prior['trials'][prior['accepted_trial']]['residual'])<1e-13
    print('RECOVER STEP',len(h),float(np.linalg.norm(r)),flush=True)
x,h,status=solve_augmented(f,x0,max_steps=30,radius=.00625,builder=builder,observer=observer,gate_policy='step')
expected=np.load(PROJECT/'results/steady_step_gate_20260920/continuation_final.npz')['x']
error=float(np.linalg.norm(x-expected));assert error<1e-12
(ROOT/'history_recovery.json').write_text(json.dumps(dict(status=status,endpoint_state_difference=error,fresh_directions=len(directions),seconds=time.perf_counter()-start,all_30_residuals_match=True),indent=2))
print('RECOVERY DONE',error,flush=True)
