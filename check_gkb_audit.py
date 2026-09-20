"""Independent saved-candidate replay and reduced-model/KKT verification."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual
from steady_gkb import svd_trust

a=json.loads((ROOT/'audit.json').read_text(encoding='utf8'))
s=(np.load(ROOT/'audit_vectors.npz') if (ROOT/'audit_vectors.npz').exists() else
   {key:value for path in sorted(ROOT.glob('vectors_*.npz')) for key,value in np.load(path).items()})
x=s['x'];r=s['r']
source=np.load(PROJECT/'results/steady_adjoint_continuation_20260920/trajectory.npz')
assert np.array_equal(x,source['states'][-1])
f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True)
np.testing.assert_allclose(f(x),r,atol=1e-13,rtol=0)
checks=[];max_kkt=0.;max_dense=0.
for t in a['trials']:
    name=t['arm'];k=t['k'];radius=t['radius'];step=s[f'{name}_{k}_{radius}']
    rr=f(x+step);actual=float((r@r-rr@rr)/2)
    assert abs(actual-t['actual'])<1e-16
    assert abs(np.linalg.norm(step)-t['step_norm'])<1e-14
    assert bool(f.feasible(x+step))==t['feasible']
    if name=='B':
        basis=s['V'][:,:k];response=s['JV'][:,:k];y=basis.T@step
        dense=float(-r@(response@y)-.5*np.linalg.norm(response@y)**2)
        max_dense=max(max_dense,abs(dense/t['prediction']-1))
        target=np.zeros(k+1);target[0]=np.linalg.norm(r);B=s['B'][:k+1,:k]
        kkt=B.T@(B@y-target)+t['lambda_value']*y
        max_kkt=max(max_kkt,float(np.linalg.norm(kkt)/np.linalg.norm(B.T@target)))
        if k==1:assert abs(t['prediction']/t['cauchy_prediction']-1)<1e-6
    passed=bool(t['feasible'] and t['prediction']>=t['cauchy_prediction']*(1-1e-6)
       and t['step_norm']<=radius*(1+1e-6) and actual>0 and t['rho']>.1
       and all(c['prediction']>0 and c['relative']<.05 for c in t['checks']))
    assert passed==t['passed']
    checks.append(dict(arm=name,k=k,radius=radius,actual=actual))
assert max_kkt<1e-8 and max_dense<1e-5
saturation=[]
for radius in a['radii']:
    rows=[t for t in a['ladder'] if t['radius']==radius]
    p=np.array([t['prediction'] for t in rows])
    assert np.all(np.diff(p)>-1e-16)
    marginal=[dict(k=k,gain=float((p[k-1]-p[k-5])/p[k-1])) for k in range(8,33,4)]
    streak=0;first=None
    for t in marginal:
        streak=streak+1 if t['gain']<.01 else 0
        if streak>=3 and first is None:first=t['k']
    saturation.append(dict(radius=radius,marginal=marginal,first_three_consecutive=first))
result=dict(candidates_replayed=len(checks),latest_endpoint_identical=True,
            maximum_gkb_kkt_relative=max_kkt,maximum_bidiagonal_dense_prediction_relative=max_dense,
            saturation=saturation,checks=checks)
(ROOT/'independent_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('INDEPENDENT CHECKS',len(checks),max_kkt,max_dense,flush=True)
