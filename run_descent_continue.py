"""Threshold-qualified S3 continuation with rolling last-three fresh directions."""
import json,time,hashlib
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual
from steady_support_lu import build_factored
from steady_augmented_outer import solve_augmented

a=json.loads((ROOT/'source_audit.json').read_text());radius=.00625
qualification=next(v for v in a['qualified'] if v['group']=='S3' and v['radius']==radius)
assert json.loads((ROOT/'candidate_replay.json').read_text())['decision_rule_verified']
source=PROJECT/'results/steady_step_gate_20260920/continuation_final.npz';x0=np.load(source)['x']
template=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz');f=physical_residual(template,gpu=True)
history=list(np.load(ROOT/'history_directions.npz')['directions'][-3:]);fresh=[];states=[];start=time.perf_counter()
def builder(f,x,r,cache):
    print('S3 BUILD',float(np.linalg.norm(r)),flush=True)
    return build_factored(f,x,r,'C',descent=cache)
def enrich(f,x,r,d,rebuilt):
    extra=np.column_stack(history[-3:])
    if rebuilt:
        history.append(d.copy());fresh.append(d.copy());states.append(x.copy())
        np.savez_compressed(ROOT/'S3_fresh_history.npz',directions=np.array(fresh),states=np.array(states))
    return extra
def observe(x,r,h):
    row=h[-1];trial=row['trials'][row['accepted_trial']] if 'accepted_trial' in row else {}
    if len(h)==1 and trial:
        expected=next(v for v in a['trials'] if v['group']=='S3' and v['radius']==radius)
        assert abs(trial['residual']-expected['residual'])<1e-12
    print('S3 STEP',len(h),'R',float(np.linalg.norm(r)),'rho',trial.get('rho'),'radius',row.get('next_radius'),flush=True)
    np.savez_compressed(ROOT/'continuation_checkpoint.npz',x=x)
    (ROOT/'continuation_progress.json').write_text(json.dumps(dict(history=h,residual=float(np.linalg.norm(r))),indent=2))
x,h,status=solve_augmented(f,x0,max_steps=30,radius=radius,builder=builder,observer=observe,gate_policy='step',enrichment=enrich)
r=f(x);np.savez_compressed(ROOT/'continuation_final.npz',x=x,output=f.output)
result=dict(group='S3',status=status,initial_residual=float(np.linalg.norm(f(x0))),residual=float(np.linalg.norm(r)),accepted_steps=sum('accepted_trial' in row for row in h),history=h,seconds=time.perf_counter()-start,qualification=qualification,selection_reason='S3 best actual decrease is within 0.5% of the global best fixed-state candidate, without repeated annulus-gradient sweeps.',input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),numerical_root=bool(np.linalg.norm(r)<1e-7),certified_stable=False)
(ROOT/'continuation.json').write_text(json.dumps(result,indent=2));print('S3 RESULT',status,result['residual'],flush=True)
