"""Independent accepted-state replay and final derivative gates for both arms."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual
checks={}
for arm in ['fresh','recycled']:
    folder=ROOT/arm;states=np.load(folder/'trajectory.npz')['states'];a=json.loads((folder/'continuation.json').read_text(encoding='utf8'))
    f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True)
    reference=np.load(PROJECT/'results/steady_pure_gkb_20260920/trajectory.npz')['states'][-1]
    np.testing.assert_array_equal(states[0],reference)
    accepted=[row for row in a['history'] if 'accepted_trial' in row];assert len(states)==len(accepted)+1
    records=[]
    for i,row in enumerate(accepted):
        x=states[i];s=states[i+1]-x;r=f(x);rr=f(states[i+1]);t=row['trials'][row['accepted_trial']]
        actual=float((r@r-rr@rr)/2);assert abs(actual-t['actual'])<1e-15
        assert f.feasible(states[i+1]) and t['passed'] and t['rho']>.1
        assert np.linalg.norm(s)<=t['radius']*(1+1e-6)
        assert t['prediction']>=t['cauchy_prediction']*(1-1e-6)
        # Larger-actual probed candidates must have failed the final directional gate.
        for trial in row['trials']:
            if trial['eligible'] and trial['actual']>t['actual']:
                assert len(trial['checks'])==3 and not trial['passed']
        derivative_checks=[]
        for h in [1e-5,3e-6,1e-6]:
            eps=h/np.linalg.norm(s);js=(f(x+eps*s)-f(x-eps*s))/(2*eps)
            pred=float(-r@js-.5*(js@js));error=abs(pred-t['prediction'])/t['prediction']
            assert pred>0 and error<.05 and actual/pred>.1
            derivative_checks.append(error)
        records.append(dict(step=i+1,actual=actual,residual=float(np.linalg.norm(rr)),prediction_relative_errors=derivative_checks))
        print('REPLAY',arm,i+1,flush=True)
    expected=-np.log(np.linalg.norm(f(states[-1]))/np.linalg.norm(f(states[0])))
    assert abs(expected-a['progress'])<1e-12
    checks[arm]=dict(accepted_replayed=len(records),maximum_prediction_relative_error=max(max(v['prediction_relative_errors']) for v in records),records=records)
(ROOT/'independent_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8');print('AB VERIFIED')
