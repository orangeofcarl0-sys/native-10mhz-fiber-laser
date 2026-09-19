"""Replay unchanged line-search trajectory; save missing directions at RT-free steps 8–11."""
import json,time,hashlib
import numpy as np
from config import ROOT,PROJECT,config
from steady_state import CavityResidual,solve

source=PROJECT/'results/steady_block_20260919'
initial=np.load(source/'input/best_pilot_larger_krylov.npz')
x=np.load(source/'spectral_support.npz')['x']
residual=CavityResidual(config('CNT_OC',.2),.125,initial['a'],.05,.8,gpu=True)
def observer(step,x,r,dx):
    if step>=8:
        np.savez_compressed(ROOT/f'original_step_{step}.npz',x=x,r=r,dx=dx,
                            template=initial['a'])
def progress(row):
    print(json.dumps(row),flush=True)
start=time.perf_counter()
x,history,status=solve(residual,x,max_steps=12,inner=60,precondition='coarse',
    coarse_cutoff=384,rebuild_every=3,progress=progress,observer=observer)
reference=json.loads((source/'wide_newton.json').read_text())
discrepancy=float(np.max(abs(np.array([v['residual'] for v in history])-
                               np.array([v['residual'] for v in reference['history']]))))
assert discrepancy<1e-7,discrepancy
result=dict(status=status,history=history,seconds=time.perf_counter()-start,
            maximum_history_difference=discrepancy,
            code_sha256={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()
                         for name in ['steady_state.py','steady_preconditioner.py','replay_steady_directions.py']})
(ROOT/'original_replay.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('REPLAY MAXIMUM HISTORY DIFFERENCE',discrepancy,flush=True)
