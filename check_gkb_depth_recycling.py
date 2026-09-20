"""Replay every candidate and verify reduced SVD/KKT and decision arithmetic."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual
from steady_gkb import svd_trust

a=json.loads((ROOT/'audit.json').read_text(encoding='utf8'));s=np.load(ROOT/'candidate_vectors.npz')
states=np.load(PROJECT/'results/steady_pure_gkb_20260920/trajectory.npz')['states']
assert np.array_equal(s['x'],states[-1]) and np.array_equal(s['old'],states[-2])
f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True)
x=s['x'];r=f(x);np.testing.assert_allclose(r,s['r'],atol=1e-13,rtol=0)
checks=[]
for t in a['trials']:
    step=s['step_'+t['name']];rr=f(x+step);actual=float((r@r-rr@rr)/2)
    assert abs(actual-t['actual'])<1e-16
    assert bool(f.feasible(x+step))==t['feasible']
    errors=[]
    for h in [1e-5,3e-6,1e-6]:
        eps=h/np.linalg.norm(step);js=(f(x+eps*step)-f(x-eps*step))/(2*eps)
        p=float(-r@js-.5*(js@js));errors.append(abs(p/t['prediction']-1))
    np.testing.assert_allclose(errors,[c['relative'] for c in t['checks']],atol=1e-8,rtol=1e-4)
    if t['name'].startswith('K') and '+' not in t['name']:
        k=t['k'];matrix=s['B'][:k+1,:k];target=np.zeros(k+1);target[0]=np.linalg.norm(r)
    else:
        tag='C' if '+' in t['name'] else 'R'
        matrix=s['small'+tag][:,1:];target=s['target'+tag]
    y,p,lam=svd_trust(matrix,target,a['radius'])
    assert abs(p/t['prediction']-1)<1e-6
    kkt=np.linalg.norm(matrix.T@(matrix@y-target)+lam*y)/max(np.linalg.norm(matrix.T@target),1e-100)
    assert kkt<1e-6
    checks.append(dict(name=t['name'],actual=actual,kkt_relative=float(kkt),maximum_prediction_error=max(errors)))
assert a['trials'][-1]['prediction']>=a['trials'][0]['prediction']*(1-1e-6)
result=dict(candidates_replayed=len(checks),latest_and_previous_states_identical=True,checks=checks,
 maximum_prediction_error=max(t['maximum_prediction_error'] for t in checks))
(ROOT/'independent_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('DEPTH CHECKED',result['candidates_replayed'],result['maximum_prediction_error'])
