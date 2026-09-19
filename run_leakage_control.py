"""Same endpoint, same full Jacobian probes, same 120-vector budget."""
import json,time
import numpy as np
from config import ROOT,PROJECT
from steady_bordered import BorderedResidual
from steady_leakage import geometry,inverse
from steady_hookstep import arnoldi
state=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
last=np.load(PROJECT/'results/steady_parameter_20260919/bordered_expanded.npz')
rfun=BorderedResidual(state,'pump',last['normal'],gpu=True,q_bounds=(-9,2))
x=last['y'];r=rfun(x)
start=time.perf_counter();data=geometry(rfun,x,r)
rows=[]
for mode,threshold in [('baseline',10),('truncate',5),('truncate',10),('truncate',20),('cap',10)]:
    pre,info=inverse(data,len(x),mode,threshold)
    def jv(v):
        h=1e-7/max(np.linalg.norm(v),1e-100)
        return (rfun(x+h*v)-r)/h
    t=time.perf_counter();h,z,y,linear=arnoldi(jv,r,pre,limit=120)
    d=z@y
    row=dict(mode=mode,threshold=threshold,krylov=len(y),arnoldi_residual=linear,
        true_residual=float(np.linalg.norm(jv(d)+r)/np.linalg.norm(r)),
        step_norm=float(np.linalg.norm(d)),seconds=time.perf_counter()-t,**info)
    rows.append(row);print(json.dumps(row),flush=True)
    (ROOT/'leakage_control.json').write_text(json.dumps(dict(rows=rows,seconds=time.perf_counter()-start),indent=2))
np.savez_compressed(ROOT/'leakage_modes.npz',singular=data[3],right=data[4][-20:],left=data[2][:,-20:])
