"""Replay full-map candidates and independently solve saved small subset models."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual
from steady_hookstep import hookstep

a=json.loads((ROOT/'history_complete.json').read_text());v=np.load(ROOT/'complete_vectors.npz');m=np.load(ROOT/'compressed_model.npz')
f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True)
x=v['x'];r=f(x);rows=[]
for row in a['trials']:
    s=v[row['group']+'_'+str(row['radius'])];rr=f(x+s);actual=float((r@r-rr@rr)/2)
    assert abs(actual-row['actual'])<1e-16
    assert abs(np.linalg.norm(rr)-row['residual'])<1e-14
    rows.append(dict(group=row['group'],radius=row['radius'],actual_difference=abs(actual-row['actual'])))
base=int(m['base_count']);max_error=0.
for audit in a['exhaustive']:
    assert len(audit['subsets'])==128
    for row in audit['subsets']:
        ids=list(range(base))+[base+i for i in row['selected']]
        _,p,_=hookstep(m['response'][:,ids],m['state'][:,ids],float(m['beta']),audit['radius'],np.ones(m['state'].shape[0]))
        max_error=max(max_error,abs(p-row['prediction']))
    for k,best in enumerate(audit['best_by_count']):
        assert best==max((row for row in audit['subsets'] if row['count']==k),key=lambda row:row['prediction'])
assert max_error<1e-16
projection_errors={}
for name,key in [('state','state'),('response','response')]:
    b=m[key][:,:-1];d=m[key][:,-1]
    c=np.linalg.lstsq(b,d,rcond=1e-12)[0]
    value=float(np.linalg.norm(d-b@c)/np.linalg.norm(d))
    projection_errors[name]=abs(value-a['novelty']['1e-12'][name]['relative'])
assert max(projection_errors.values())<1e-10
result=dict(candidate_count=len(rows),all_replayed=True,initial_difference=float(np.linalg.norm(r-v['residual'])),
    exhaustive_models_replayed=384,model_prediction_max_difference=max_error,projection_crosscheck=projection_errors,rows=rows)
(ROOT/'independent_checks.json').write_text(json.dumps(result,indent=2))
print(result)
