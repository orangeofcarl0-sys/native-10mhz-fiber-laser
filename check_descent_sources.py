"""Independent replay of the twelve fixed-state candidates and decision rule."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual

a=json.loads((ROOT/'source_audit.json').read_text());v=np.load(ROOT/'audit_vectors.npz')
x=np.load(PROJECT/'results/steady_step_gate_20260920/continuation_final.npz')['x']
template=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
f=physical_residual(template,gpu=True);r=f(x);checks=[]
for row in a['trials']:
    s=v[row['group']+'_'+str(row['radius'])]
    probe=physical_residual(template,{a['selected_parameter']:float(s[-1])},gpu=True) if row['group']=='S4' else f
    xx=x+(s[:-1] if row['group']=='S4' else s);rr=probe(xx)
    difference=abs(float(np.linalg.norm(rr))-row['residual']);assert difference<1e-14
    actual=float((r@r-rr@rr)/2);assert abs(actual-row['actual_reduction'])<1e-20
    checks.append(dict(group=row['group'],radius=row['radius'],endpoint_difference=difference,feasible=bool(probe.feasible(xx))))
    assert probe.feasible(xx)
expected=[(row['group'],row['radius']) for row in a['trials'] if row['group']!='S1' and row['passed'] and row['actual_gain_over_S1']>=1.25 and next(b for b in a['trials'] if b['group']=='S1' and b['radius']==row['radius'])['passed']]
assert expected==[(row['group'],row['radius']) for row in a['qualified']]
(ROOT/'candidate_replay.json').write_text(json.dumps(dict(checks=checks,decision_rule_verified=True),indent=2));print('All twelve candidates and qualification decisions verified',flush=True)
