"""At a rejected state, compare forward and centered directional derivatives."""
import json,time
import numpy as np
from config import ROOT,PROJECT
from steady_parameters import parameter_residual
from steady_support_lu import build_factored
from steady_hookstep import arnoldi
state=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
pilot=ROOT/'forward_quality_gate'
metadata=json.loads((pilot/'anchor.json').read_text());saved=np.load(pilot/'anchor.npz')
x=saved['x'];residual=parameter_residual(state,'pump',(metadata['pump_W']-.05)/.005,gpu=True);r=residual(x)
pre,info=build_factored(residual,x,r,'C');rows=[]
for method,epsilon in [('forward',1e-7),('central',1e-6),('central',1e-5)]:
    def jv(v):
        h=epsilon/max(np.linalg.norm(v),1e-100)
        return ((residual(x+h*v)-r)/h if method=='forward' else (residual(x+h*v)-residual(x-h*v))/(2*h))
    start=time.perf_counter();hh,z,y,predicted=arnoldi(jv,r,pre,limit=240,tolerance=.008);d=z@y
    checks=[]
    for step in [1e-4,1e-5,1e-6]:
        h=step/max(np.linalg.norm(d),1e-100);jd=(residual(x+h*d)-residual(x-h*d))/(2*h)
        checks.append(dict(epsilon=step,relative=float(np.linalg.norm(r+jd)/np.linalg.norm(r))))
    row=dict(method=method,epsilon=epsilon,krylov=len(y),predicted=predicted,step_norm=float(np.linalg.norm(d)),
        own_relative=float(np.linalg.norm(r+jv(d))/np.linalg.norm(r)),checks=checks,seconds=time.perf_counter()-start)
    rows.append(row);np.savez_compressed(ROOT/(method+'_'+str(epsilon)+'_direction.npz'),direction=d)
    (ROOT/'jvp_consistency.json').write_text(json.dumps(dict(state='anchor.npz from forward-quality-gate pilot',physical_residual=float(np.linalg.norm(r)),rows=rows,coarse=info),indent=2))
    print(json.dumps(row),flush=True)
