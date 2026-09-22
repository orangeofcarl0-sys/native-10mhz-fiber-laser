"""Independent replay and merit-defect identity; no derivative calls."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual
v={}
for path in ([ROOT/'candidates.npz'] if (ROOT/'candidates.npz').exists() else sorted(ROOT.glob('candidates_part*.npz'))):
    with np.load(path) as chunk:v.update({key:chunk[key] for key in chunk.files})
a=json.loads((ROOT/'radius_map.json').read_text(encoding='utf8'));x=v['x'];r=v['r']
f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True)
maximum=0.;identity=0.;checks=[]
for row in a['rows']:
    name=row['name'];s=v[name+'_step'];js=v[name+'_js'];rr=f(x+s)
    maximum=max(maximum,float(np.linalg.norm(rr-v[name+'_residual'])))
    actual=.5*(r@r-rr@rr);prediction=-r@js-.5*(js@js);e=rr-r-js
    cross=float(-(r+js)@e);penalty=float(-.5*(e@e))
    identity=max(identity,abs(actual-prediction-cross-penalty))
    assert abs(actual-row['actual'])<1e-15
    assert abs(np.linalg.norm(e)/np.linalg.norm(js)-row['nonlinear_defect'])<1e-7
    assert bool(f.feasible(x+s))==row['feasible']
    checks.append(dict(name=name,cross_term=cross,defect_penalty=penalty,identity_error=float(abs(actual-prediction-cross-penalty))))
assert maximum<1e-12 and identity<1e-18 and len(checks)==65
out=dict(candidates_replayed=len(checks),max_residual_difference=maximum,max_identity_error=identity,derivative_calls=0,checks=checks)
(ROOT/'independent_checks.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print('VERIFIED',len(checks),maximum,identity)
