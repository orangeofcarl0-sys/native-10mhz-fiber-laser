"""Independent endpoint replay and acceptance audit for paired runs."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_parameters import parameter_residual

template=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
checks={}
for arm in ['A','B']:
    result=json.loads((ROOT/(arm+'.json')).read_text());x=np.load(ROOT/(arm+'_final.npz'))['x']
    f=parameter_residual(template,'pump',(result['protocol']['pump_W']-.05)/.005,gpu=True)
    initial=np.load(PROJECT/result['protocol']['input'])['x'];ri=f(initial)
    r=f(x);rows=[h for h in result['history'] if 'accepted_trial' in h]
    selected=[h['trials'][h['accepted_trial']] for h in rows]
    expected=result['initial_residual'];chain=True
    for h,t in zip(rows,selected):
        chain &= abs(h['residual']-expected)<1e-14
        expected=t['residual']
    checks[arm]=dict(recomputed_residual=float(np.linalg.norm(r)),
        endpoint_difference=float(abs(np.linalg.norm(r)-result['physical_residual'])),
        accepted_history_continuous=bool(chain),accepted_count_matches=len(rows)==result['accepted_steps'],
        accepted_linear_gate=all(h['linear_gate']<.01 for h in rows),
        actual_merit_monotone=all(t['actual_reduction']>0 for t in selected),
        independent_acceptance=all(min(t['rho'],t['true_rho'])>.1 for t in selected),
        descent_verified=all(max(h['descent_slope'],h['checked_descent_slope'])<0 for h in rows) if arm=='B' else None,
        full_rebuild_directions=sum(h.get('gradient_source')=='full_output_rebuild' for h in rows),
        cheap_directions=sum(h.get('gradient_source')=='cheap_current_residual' for h in rows),
        model_fallbacks=sum(t['kind']=='cauchy_model_fallback' for t in selected),
        actual_fallbacks=sum(t['kind']=='cauchy_actual_fallback' for t in selected),
        raw_model_failures=sum(not t['guard']['raw_model_pass'] for h in result['history'] for t in h['trials'] if 'guard' in t),
        initial_blocks=dict(field=float(np.linalg.norm(ri[:4*f.n])),population=float(np.linalg.norm(ri[4*f.n:-2])),gauge=float(np.linalg.norm(ri[-2:]))),
        residual_blocks=dict(field=float(np.linalg.norm(r[:4*f.n])),population=float(np.linalg.norm(r[4*f.n:-2])),gauge=float(np.linalg.norm(r[-2:]))))
    for key in ['accepted_history_continuous','accepted_count_matches','accepted_linear_gate','actual_merit_monotone','independent_acceptance']:
        assert checks[arm][key],(arm,key)
    if arm=='B':assert checks[arm]['descent_verified']
    assert checks[arm]['endpoint_difference']<1e-14
(ROOT/'endpoint_checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
