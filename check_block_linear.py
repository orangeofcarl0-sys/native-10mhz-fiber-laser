"""Same-state right/left Krylov comparison, including true residual checks."""
import os, json, time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres
from config import ROOT, config
from steady_state import CavityResidual
from steady_preconditioner import build

saved = np.load(Path(os.environ['LASER_STEADY_DIR'])/'best_pilot_larger_krylov.npz')
a,p = saved['a'],saved['pop']
residual = CavityResidual(config('CNT_OC',.2),.125,a,.05,.8,gpu=True)
x = residual.pack(a,p)
r = residual(x); m = 2*a.shape[-1]
b = a/residual.scale+(r[:m]+1j*r[m:2*m]).reshape(a.shape)
x[-2] = np.angle(np.vdot(a,b)); r=residual(x)
def jv(v):
    h=1e-7/max(np.linalg.norm(v),1e-100)
    return (residual(x+h*v)-r)/h
jacobian=LinearOperator((x.size,x.size),matvec=jv,dtype=float)
preconditioner,diagnostics=build(residual,x,r)
rows=[]
for side in ['none','left','right']:
    start=time.perf_counter(); calls=residual.evaluations
    operator=jacobian if side!='right' else LinearOperator(jacobian.shape,
                           matvec=lambda v:jacobian@(preconditioner@v),dtype=float)
    d,info=lgmres(operator,-r,M=preconditioner if side=='left' else None,
                  rtol=.03,atol=0,maxiter=2,inner_m=60,outer_k=3)
    dx=preconditioner@d if side=='right' else d
    error=np.linalg.norm(jv(dx)+r)/np.linalg.norm(r)
    rows.append(dict(side=side,info=int(info),relative_linear_residual=float(error),
                     calls=residual.evaluations-calls,seconds=time.perf_counter()-start))
    print(json.dumps(rows[-1]),flush=True)
    np.savez_compressed(ROOT/f'linear_{side}.npz',x=x,dx=dx)
(ROOT/'linear_sides.json').write_text(json.dumps(dict(diagnostics=diagnostics,trials=rows),indent=2),encoding='utf8')
