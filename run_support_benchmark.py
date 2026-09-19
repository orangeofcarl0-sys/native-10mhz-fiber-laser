"""Single fixed state, one Arnoldi sequence per support, budget prefixes."""
import json,time,hashlib
import numpy as np
from config import ROOT,PROJECT
from steady_parameters import parameter_residual
from steady_support import build_support
from steady_hookstep import arnoldi
state=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
prior=PROJECT/'results/steady_energy_20260919'
last=np.load(prior/'fixed_final.npz');pump=json.loads((prior/'fixed_final.json').read_text())['pump_W']
residual=parameter_residual(state,'pump',(pump-.05)/.005,gpu=True)
x=last['x'];r=residual(x);rows=[]
for label in ['A','B','C']:
    start=time.perf_counter();print('BUILD',label,flush=True)
    pre,info=build_support(residual,x,r,label);build_seconds=time.perf_counter()-start
    def jv(v):
        h=1e-7/max(np.linalg.norm(v),1e-100)
        return (residual(x+h*v)-r)/h
    start=time.perf_counter();h,z,y,linear=arnoldi(jv,r,pre,limit=240,tolerance=0.)
    for requested in [120,180,240]:
        k=min(requested,z.shape[1]);target=np.zeros(k+1);target[0]=np.linalg.norm(r)
        coeff=np.linalg.lstsq(h[:k+1,:k],target,rcond=1e-12)[0];d=z[:,:k]@coeff
        eps=1e-6/max(np.linalg.norm(d),1e-100)
        central=(residual(x+eps*d)-residual(x-eps*d))/(2*eps)
        row=dict(support=label,budget=requested,krylov=k,
            arnoldi_relative=float(np.linalg.norm(h[:k+1,:k]@coeff-target)/np.linalg.norm(r)),
            true_relative=float(np.linalg.norm(r+jv(d))/np.linalg.norm(r)),
            central_relative=float(np.linalg.norm(r+central)/np.linalg.norm(r)),
            step_norm=float(np.linalg.norm(d)),build_seconds=build_seconds,sequence_seconds=time.perf_counter()-start,**info)
        rows.append(row);print(json.dumps(row),flush=True)
    (ROOT/'support_benchmark.json').write_text(json.dumps(dict(pump_W=pump,initial_residual=float(np.linalg.norm(r)),
        input='results/steady_energy_20260919/fixed_final.npz',rows=rows,
        source_sha256={n:hashlib.sha256((PROJECT/n).read_bytes()).hexdigest() for n in ['steady_hookstep.py','steady_support.py','run_support_benchmark.py']}),indent=2))
