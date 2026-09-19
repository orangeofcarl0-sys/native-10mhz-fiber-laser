"""Independent final-state replay and step-gate history checks."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_parameters import parameter_residual
b=json.loads((ROOT/'continuation.json').read_text())
f=parameter_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),'pump',(.0275-.05)/.005,gpu=True)
x=np.load(ROOT/'continuation_final.npz')['x'];r=f(x)
rows=[v for v in b['history'] if 'accepted_trial' in v];trials=[v['trials'][v['accepted_trial']] for v in rows]
expected=b['initial_residual']
for row,t in zip(rows,trials):
    assert abs(row['residual']-expected)<1e-14;expected=t['residual']
    assert t['candidate_model_pass'] and t['actual_reduction']>0 and min(t['rho'],t['true_rho'])>.1
    assert all(q>0 and abs(q-t['prediction'])<.05*abs(t['prediction']) for q in t['checked_predictions'])
    assert t['G_C']>=1-1e-6
    assert max(row['descent_slope'],row['checked_descent_slope'])<0
    if row['newton_norm']<=row['radius']:assert row['linear_gate']<.01
assert len(rows)==b['accepted_steps']
assert abs(expected-b['residual'])<1e-14
assert abs(np.linalg.norm(r)-b['residual'])<1e-14
result=dict(recomputed_residual=float(np.linalg.norm(r)),endpoint_difference=float(abs(np.linalg.norm(r)-b['residual'])),all_acceptance_checks_pass=True,accepted=len(rows),previously_blocked_but_accepted=sum(row['linear_gate']>=.01 for row in rows),fresh=sum(row['gradient_source']=='full_output_rebuild' for row in rows),cheap=sum(row['gradient_source']=='cheap_current_residual' for row in rows),residual_blocks=dict(field=float(np.linalg.norm(r[:4*f.n])),population=float(np.linalg.norm(r[4*f.n:-2])),gauge=float(np.linalg.norm(r[-2:]))),energy_nJ=float(np.sum(abs(f.output)**2)*f.dt/1000))
(ROOT/'endpoint_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
