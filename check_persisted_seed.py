"""Independent full-map trajectory, immutable seed and recorded policy audit."""
import hashlib,json
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual
from steady_seed_bank import trigger_diagnostic

a=json.loads((ROOT/'continuation.json').read_text());v=np.load(ROOT/'trajectory.npz')
f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True)
saved=np.load(PROJECT/'results/steady_active_history_20260920/active_vectors.npz')
assert np.array_equal(v['seed'],saved['annulus'])
assert hashlib.sha256(v['seed'].tobytes()).hexdigest()==a['seed_sha256']
accepted=[r for r in a['history'] if 'accepted_trial' in r]
assert len(v['states'])==len(accepted)+1
norms=[float(np.linalg.norm(f(x))) for x in v['states']];errors=[];slope_errors=[]
for i,row in enumerate(accepted):
    t=row['trials'][row['accepted_trial']]
    errors.append(abs(norms[i+1]-t['residual']))
    assert abs(norms[i]-row['residual'])<1e-13
    assert row['history_bank_size']<=12
    current=v['states'][i];rr=f(current);epsilon=1e-5/np.linalg.norm(v['seed'])
    response=(f(current+epsilon*v['seed'])-f(current-epsilon*v['seed']))/(2*epsilon)
    slope_errors.append(abs(float(rr@response)-row['seed_slope']))
    assert t['candidate_model_pass'] and min(t['rho'],t['true_rho'])>.1
    assert t['actual_reduction']>0
    assert abs(.5*(norms[i]**2-norms[i+1]**2)-t['actual_reduction'])<1e-16
    for event in row['bank_pruning']:
        selected=next(e for e in event['trials'] if e['index']==event['removed'])
        minimum=min(e['loss'] for e in event['trials'])
        assert selected['loss']<=minimum+1e-10*abs(event['full_prediction'])
    assert row['trigger_diagnostic']==trigger_diagnostic(a['history'][:row['step']+1])
assert max(errors,default=0)<1e-13
assert max(slope_errors,default=0)<1e-13
last=f(v['states'][-1]);output=f.output.copy();final=np.load(ROOT/'continuation_final.npz')
assert np.array_equal(final['x'],v['states'][-1])
np.testing.assert_allclose(output,final['output'],rtol=1e-13,atol=1e-13)
result=dict(accepted_states_replayed=len(accepted)+1,max_residual_difference=max(errors,default=0),
    output_relative_difference=float(np.linalg.norm(output-final['output'])/np.linalg.norm(output)),
    seed_slope_max_difference=max(slope_errors,default=0),
    immutable_seed_verified=True,bank_capacity_and_deletions_verified=True,diagnostic_triggers_verified=True)
(ROOT/'independent_checks.json').write_text(json.dumps(result,indent=2));print(result)
