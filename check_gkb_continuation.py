"""Independently replay accepted states and three-scale step predictions."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual

a=json.loads((ROOT/'continuation.json').read_text(encoding='utf8'))
states=np.load(ROOT/'trajectory.npz')['states']
source=np.load(PROJECT/'results/steady_adjoint_continuation_20260920/trajectory.npz')['states'][-1]
assert np.array_equal(states[0],source)
f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True)
residuals=[f(x) for x in states]
np.testing.assert_allclose([np.linalg.norm(r) for r in residuals],a['norms'],atol=1e-13,rtol=0)
accepted=[row for row in a['history'] if 'accepted_trial' in row]
assert len(states)==len(accepted)+1
checked=[]
for i,row in enumerate(accepted):
    x=states[i];step=states[i+1]-x;r=residuals[i];rr=residuals[i+1]
    t=row['trials'][row['accepted_trial']]
    actual=float((r@r-rr@rr)/2)
    assert abs(actual-t['actual'])<1e-16 and actual>0
    assert f.feasible(x+step) and t['rho']>.1 and t['passed']
    assert np.linalg.norm(step)<=t['radius']*(1+1e-6)
    assert t['prediction']>=t['cauchy_prediction']*(1-1e-6)
    errors=[]
    for h in [1e-5,3e-6,1e-6]:
        eps=h/np.linalg.norm(step)
        js=(f(x+eps*step)-f(x-eps*step))/(2*eps)
        p=float(-r@js-.5*(js@js));error=abs(p/t['prediction']-1)
        assert error<.05 and p>0
        errors.append(error)
    checks=[q for q in row['ladder'] if 'prediction' in q]
    for j,q in enumerate(checks):
        if j:
            gain=(q['prediction']-checks[j-1]['prediction'])/q['prediction']
            assert abs(gain-q['marginal_gain'])<1e-12
        if j>=2 and all(v['marginal_gain'] is not None and 0<=v['marginal_gain']<.01 for v in checks[j-1:j+1]):
            assert j==len(checks)-1 and row['saturated']
    assert row['k']<=64 and row['u_orth']<1e-10 and row['v_orth']<1e-10
    checked.append(dict(step=i+1,actual=actual,maximum_prediction_error=max(errors)))
result=dict(states_replayed=len(states),steps_verified=len(accepted),
            maximum_prediction_error=max([t['maximum_prediction_error'] for t in checked],default=0),checks=checked)
(ROOT/'independent_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('CHECKED',len(states),len(accepted),result['maximum_prediction_error'])
