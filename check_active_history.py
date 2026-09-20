"""Independent full-cavity replay of the fifteen saved active-history candidates."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual

a=json.loads((ROOT/'active_history.json').read_text());v=np.load(ROOT/'active_vectors.npz')
f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True)
x=v['x'];r=f(x);rows=[]
for row in a['trials']:
    s=v[row['group']+'_'+str(row['radius'])];rr=f(x+s)
    actual=float((r@r-rr@rr)/2)
    assert abs(actual-row['actual'])<1e-16
    assert abs(np.linalg.norm(rr)-row['residual'])<1e-14
    rows.append(dict(group=row['group'],radius=row['radius'],actual_difference=abs(actual-row['actual'])))
for audit in a['greedy']:
    selected=[];previous=None
    for level in audit['ladder']:
        winner=max(level['candidates'],key=lambda c:c['prediction'])
        assert winner['index'] not in selected
        selected.append(winner['index']);assert selected==level['selected']
        assert winner['prediction']==level['prediction']
        if previous is not None:assert level['prediction']>=previous-1e-16
        previous=level['prediction']
    expected=next((i for i,z in enumerate(audit['ladder']) if z['marginal_gain']<.05),4)
    assert audit['policy_count']==expected
for name,key in [('D','history'),('JD','history_responses')]:
    s=np.linalg.svd(v[key].T,compute_uv=False)
    np.testing.assert_allclose(s,a['rank'][name]['singular'],rtol=1e-12)
result=dict(candidate_count=len(rows),all_replayed=True,initial_difference=float(np.linalg.norm(r-v['residual'])),
            greedy_selection_verified=True,rank_verified=True,rows=rows)
(ROOT/'independent_checks.json').write_text(json.dumps(result,indent=2))
print(result)
