"""Conditional 30-step continuation with independently checked selected steps."""
import json,time,hashlib
import numpy as np
from config import ROOT,PROJECT
from steady_parameters import parameter_residual
from steady_augmented_outer import solve_augmented
from steady_support_lu import build_factored
assert json.loads((ROOT/'audit.json').read_text())['continuation_allowed']
source=PROJECT/'results/steady_augmented_pair_20260920/B_final.npz'
x0=np.load(source)['x'];template=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
f=parameter_residual(template,'pump',(.0275-.05)/.005,gpu=True)
start=time.perf_counter()
def builder(f,x,r,cache):
    print('BUILD',float(np.linalg.norm(r)),flush=True)
    return build_factored(f,x,r,'C',descent=cache)
def observer(x,r,h):
    row=h[-1];t=row['trials'][row['accepted_trial']] if 'accepted_trial' in row else {}
    print('STEP',row['step'],'R',float(np.linalg.norm(r)),'rho',t.get('rho'),'radius',row.get('next_radius'),flush=True)
    np.savez_compressed(ROOT/'continuation_checkpoint.npz',x=x)
    (ROOT/'progress.json').write_text(json.dumps(dict(history=h,residual=float(np.linalg.norm(r))),indent=2))
x,h,status=solve_augmented(f,x0,max_steps=30,radius=.00625,builder=builder,observer=observer,gate_policy='step')
r=f(x)
np.savez_compressed(ROOT/'continuation_final.npz',x=x,output=f.output)
result=dict(status=status,initial_residual=h[0]['residual'],residual=float(np.linalg.norm(r)),accepted_steps=sum('accepted_trial' in row for row in h),history=h,seconds=time.perf_counter()-start,numerical_root=bool(np.linalg.norm(r)<1e-7),certified_stable=False,input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
(ROOT/'continuation.json').write_text(json.dumps(result,indent=2));print('RESULT',status,result['residual'],flush=True)
