"""Independent cavity-map curvature and saved linear-response replay."""
import json
import numpy as np
from config import ROOT,PROJECT
from discrete_adjoint import DiscreteAdjoint
from steady_descent_sources import physical_residual

data=json.loads((ROOT/'geometry.json').read_text(encoding='utf8'))
v=np.load(ROOT/'vectors.npz');x=v['x'];r=v['r'];g=v['g']
f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True)
adj=DiscreteAdjoint(f)
assert np.linalg.norm(f(x)-r)<1e-13
assert np.linalg.norm(adj.value_and_vjp(x)[1]-g)<1e-12
checks=dict(curvature=[],linear=[],symmetry=[])
for row in data['curvature']:
    name=row['name'];d=v[name+'_direction'];ref=row['scales'][2];scales=[]
    for h in [1e-4,3e-5,1e-5,3e-6]:
        rp=f(x+h*d);rm=f(x-h*d)
        full=float((rp@rp-2*r@r+rm@rm)/(2*h*h))
        missing=float(r@(rp-2*r+rm)/(h*h))
        scales.append(dict(h=h,full=full,missing=missing,
            full_error_over_gn=abs(full-ref['full'])/ref['gn'],
            missing_error_over_gn=abs(missing-ref['missing'])/ref['gn']))
    # Second differences lose precision at tiny h; retain the complete ladder.
    best=min(max(t['full_error_over_gn'],t['missing_error_over_gn']) for t in scales)
    checks['curvature'].append(dict(name=name,scales=scales,best_joint_error=best))
    assert best<.01,(name,best)
names=[row['name'] for row in data['curvature']]
for i,a in enumerate(names):
    for b in names[i+1:]:
        left=float(v[a+'_direction']@v[b+'_hv']);right=float(v[b+'_direction']@v[a+'_hv'])
        scale=np.linalg.norm(v[a+'_hv'])*np.linalg.norm(v[b+'_direction'])+np.linalg.norm(v[b+'_hv'])*np.linalg.norm(v[a+'_direction'])
        error=abs(left-right)/max(scale,1e-100)
        checks['symmetry'].append(dict(a=a,b=b,left=left,right=right,operator_scaled_error=error))
        assert error<1e-4,(a,b,error)
for row in data['linear']:
    k=row['k'];s=v[f'linear_step_{k}'];lr=v[f'linear_residual_{k}'];errors=[]
    for h in [1e-5,3e-6,1e-6]:
        eps=h/np.linalg.norm(s);response=(f(x+eps*s)-f(x-eps*s))/(2*eps)
        errors.append(float(np.linalg.norm(response-(lr-r))/np.linalg.norm(lr-r)))
    normal=float(np.linalg.norm(adj.value_and_vjp(x,lr)[1])/np.linalg.norm(g))
    assert max(errors)<.001,(k,errors)
    assert abs(normal-row['full_normal_ratio'])<1e-8
    checks['linear'].append(dict(k=k,three_scale_response_errors=errors,full_normal_ratio=normal))
checks['passed']=True
(ROOT/'independent_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
print('INDEPENDENT CHECKS PASS',len(checks['curvature']),len(checks['linear']),len(checks['symmetry']),flush=True)
