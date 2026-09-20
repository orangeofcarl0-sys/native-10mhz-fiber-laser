"""Independent full-map and saved compressed-model replay at all four states."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual
from steady_hookstep import hookstep
a=json.loads((ROOT/'fresh_retrospective.json').read_text())
assert [s['step'] for s in a['states']]==[5,11,20,27]
f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True)
checks=[]
for row in a['states']:
    v=np.load(ROOT/f"state_{row['step']}.npz");m=np.load(ROOT/f"model_{row['step']}.npz")
    x=v['x'];r=f(x);base=int(m['base_count'])
    for trial in row['trials']:
        group=trial['group'];s=v['step_'+group];rr=f(x+s);actual=float((r@r-rr@rr)/2)
        ids=list(range(base))+({'history':[],'old':[base],'fresh':[base+1]}[group])
        _,p,_=hookstep(m['response'][:,ids],m['state'][:,ids],float(m['beta']),row['radius'],np.ones(m['state'].shape[0]))
        assert abs(p-trial['prediction'])<1e-16 and abs(actual-trial['actual'])<1e-16
        checks.append(dict(step=row['step'],group=group,actual_difference=abs(actual-trial['actual']),prediction_difference=abs(p-trial['prediction'])))
    d=v['fresh'];eps=1e-5/np.linalg.norm(d);j=(f(x+eps*d)-f(x-eps*d))/(2*eps)
    np.testing.assert_allclose(j,v['jf'],rtol=1e-10,atol=1e-10)
    assert abs(float(-r@j)-row['fresh_descent'])<1e-13
    assert abs(float(v['old']@d)-row['cos_x'])<1e-13
    assert abs(float(v['jo']@j/(np.linalg.norm(v['jo'])*np.linalg.norm(j)))-row['cos_J'])<1e-12
count=sum(s['Gold']<1.1 and s['Gfresh']>1.5 for s in a['states'])
assert count==a['threshold_count']
assert len(checks)==12
result=dict(states=4,candidates_replayed=len(checks),direction_diagnostics_verified=True,threshold_count=count,checks=checks)
(ROOT/'independent_checks.json').write_text(json.dumps(result,indent=2));print(result)
