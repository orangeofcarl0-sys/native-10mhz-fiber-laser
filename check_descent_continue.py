"""Independent S3 endpoint, accepted-step and archived-output validation."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual
b=json.loads((ROOT/'continuation.json').read_text());state=np.load(ROOT/'continuation_final.npz')
f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True)
r=f(state['x']);expected=b['initial_residual'];rows=[v for v in b['history'] if 'accepted_trial' in v]
for row in rows:
    t=row['trials'][row['accepted_trial']]
    assert abs(row['residual']-expected)<1e-14;expected=t['residual']
    assert row['extra_direction_count']==3 and t['candidate_model_pass']
    assert t['actual_reduction']>0 and min(t['rho'],t['true_rho'])>.1 and t['G_C']>=1-1e-6
    assert all(q>0 and abs(q-t['prediction'])<.05*abs(t['prediction']) for q in t['checked_predictions'])
assert len(rows)==b['accepted_steps'];assert abs(expected-b['residual'])<1e-14
assert abs(np.linalg.norm(r)-b['residual'])<1e-14
output_error=float(np.linalg.norm(f.output-state['output']));assert output_error<1e-12
fresh_steps=[v['step']+1 for v in b['history'] if v.get('extra_direction_count')==3 and v.get('precondition_rebuilt')]
assert len(np.load(ROOT/'S3_fresh_history.npz')['directions'])==len(fresh_steps)
trials=[v['trials'][v['accepted_trial']] for v in rows]
result=dict(fresh_source_outer_steps=fresh_steps,accepted=len(rows),all_acceptance_checks_pass=True,endpoint_difference=float(abs(np.linalg.norm(r)-b['residual'])),output_difference=output_error,fresh=sum(v['precondition_rebuilt'] for v in rows),cheap=sum(not v['precondition_rebuilt'] for v in rows),fallbacks=sum(t['kind']!='augmented' for t in trials),residual_blocks=dict(field=float(np.linalg.norm(r[:4*f.n])),population=float(np.linalg.norm(r[4*f.n:-2])),gauge=float(np.linalg.norm(r[-2:]))),output_energy_nJ=float(np.sum(abs(f.output)**2)*f.dt/1000))
(ROOT/'endpoint_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
