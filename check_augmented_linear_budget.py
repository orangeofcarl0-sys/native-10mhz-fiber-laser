"""Check failed paired endpoints with the full 240-vector budget (no state step)."""
import json,hashlib
import numpy as np
from config import ROOT,PROJECT
from steady_parameters import parameter_residual
from steady_support_lu import build_factored
from steady_hookstep import arnoldi
from steady_diagnostics import blocks

template=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz');rows=[]
for arm in ['A','B']:
    run=json.loads((ROOT/(arm+'.json')).read_text())
    if run['status']!='linear_accuracy_limited':continue
    last=run['history'][-1]
    if last['linear_attempts'][-1]['dimension']==240:
        v=last['linear_attempts'][-1]
        rows.append(dict(arm=arm,dimension=240,model=v['model'],independent=v['independent'],
            gate_pass=bool(v['independent']<.01),reused_original_full_budget_check=True,
            newton_norm=last['newton_norm']))
        continue
    x=np.load(ROOT/(arm+'_final.npz'))['x']
    f=parameter_residual(template,'pump',(run['protocol']['pump_W']-.05)/.005,gpu=True);r=f(x)
    print('FULL_BUDGET_BUILD',arm,flush=True);pre,info=build_factored(f,x,r,'C')
    def jv(v,step=1e-5):
        eps=step/max(np.linalg.norm(v),1e-100)
        return (f(x+eps*v)-f(x-eps*v))/(2*eps)
    h,z,y,linear=arnoldi(jv,r,pre,limit=240,tolerance=0.)
    d=z@y;independent=float(np.linalg.norm(r+jv(d,1e-6))/np.linalg.norm(r))
    row=dict(arm=arm,original_attempts=run['history'][-1]['linear_attempts'],dimension=len(y),
        model=linear,independent=independent,newton_norm=float(np.linalg.norm(d)),
        gate_pass=bool(independent<.01),coarse=info,newton_blocks=blocks(f,d),
        endpoint_sha256=hashlib.sha256((ROOT/(arm+'_final.npz')).read_bytes()).hexdigest())
    np.savez_compressed(ROOT/(arm+'_full_budget_direction.npz'),newton=d)
    rows.append(row);print('FULL_BUDGET_RESULT',json.dumps(row),flush=True)
(ROOT/'linear_budget_check.json').write_text(json.dumps(rows,indent=2))
