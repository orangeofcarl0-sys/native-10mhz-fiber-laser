"""Independent trajectory/gradient/guard replay after full-adjoint integration."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual
from discrete_adjoint import DiscreteAdjoint
from steady_cauchy import cauchy_step

a=json.loads((ROOT/'continuation.json').read_text());v=np.load(ROOT/'trajectory.npz')
f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True)
adj=DiscreteAdjoint(f);accepted=[row for row in a['history'] if 'accepted_trial' in row]
assert len(v['states'])==len(accepted)+1==len(v['directions'])+1
prior=np.load(PROJECT/'results/steady_persisted_seed_20260920/trajectory.npz')
assert np.array_equal(v['states'][0],prior['states'][-1])
checks=[]
for i,row in enumerate(accepted):
    assert row['gradient_source']=='full_adjoint'
    x=v['states'][i];r=f(x);_,g=adj.value_and_vjp(x);d=-g/np.linalg.norm(g)
    t=row['trials'][row['accepted_trial']];rr=f(v['states'][i+1])
    err=float(np.linalg.norm(d-v['directions'][i]));assert err<1e-10
    assert abs(np.linalg.norm(g)/row['gradient_norm']-1)<1e-10
    assert abs(np.linalg.norm(r)-row['residual'])<1e-13
    assert abs(np.linalg.norm(rr)-t['residual'])<1e-13
    jd=(f(x+1e-5*d)-f(x-1e-5*d))/(2e-5)
    gradient_error=abs(float(-r@jd)/np.linalg.norm(g)-1)
    assert gradient_error<1e-4
    _,pc,alpha=cauchy_step(r,d,jd,t['radius'],np.ones_like(x))
    assert abs(pc/t['cauchy_prediction']-1)<1e-6
    assert t['prediction']>=pc*(1-1e-6)
    assert t['candidate_model_pass'] and min(t['rho'],t['true_rho'])>.1 and t['actual_reduction']>0
    assert abs((r@r-rr@rr)/2-t['actual_reduction'])<1e-16
    assert row['history_bank_size']<=12
    assert row['full_history_count']==sum(q.startswith('full_') for q in row['history_ids'])
    assert row['old_history_count']+row['full_history_count']==row['history_bank_size']
    for event in row['bank_pruning']:
        selected=next(z for z in event['trials'] if z['index']==event['removed'])
        assert selected['loss']<=min(z['loss'] for z in event['trials'])+1e-10*abs(event['full_prediction'])
    checks.append(dict(step=i,residual=float(np.linalg.norm(rr)),direction_error=err,
                       cauchy_relative_error=abs(pc/t['cauchy_prediction']-1),
                       gradient_directional_relative_error=float(gradient_error)))
    print('CHECKED',i,flush=True)
final=np.load(ROOT/'final.npz');f(v['states'][-1])
np.testing.assert_allclose(f.output,final['output'],rtol=1e-13,atol=1e-13)
gains=[1-r['trials'][r['accepted_trial']]['residual']/r['residual'] for r in accepted]
assert abs(float(np.mean(gains[-5:]))-a['last_five_mean_norm_gain'])<1e-15
result=dict(states_replayed=len(v['states']),current_full_directions_verified=len(checks),
            guards_and_bank_rule_verified=True,latest_endpoint_verified=True,checks=checks)
(ROOT/'independent_checks.json').write_text(json.dumps(result,indent=2));print('REPLAY COMPLETE')
