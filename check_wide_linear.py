"""Broad contiguous Fourier Jacobian including residual-bearing sidebands."""
import os,json,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator,lgmres
from config import ROOT,config
from steady_state import CavityResidual
from steady_preconditioner import build_coarse

saved=np.load(Path(os.environ['LASER_STEADY_DIR'])/'best_pilot_larger_krylov.npz')
residual=CavityResidual(config('CNT_OC',.2),.125,saved['a'],.05,.8,gpu=True)
x=np.load(ROOT/'linear_none.npz')['x'];r=residual(x)
def jv(v):
    h=1e-7/max(np.linalg.norm(v),1e-100)
    return (residual(x+h*v)-r)/h
start=time.perf_counter();calls=residual.evaluations
preconditioner,diagnostics=build_coarse(residual,x,r,cutoff=384)
construction_calls=residual.evaluations-calls
operator=LinearOperator((len(x),len(x)),matvec=lambda v:jv(preconditioner@v),dtype=float)
d,info=lgmres(operator,-r,rtol=.03,atol=0,maxiter=2,inner_m=60,outer_k=3)
dx=preconditioner@d
error=np.linalg.norm(jv(dx)+r)/np.linalg.norm(r)
row=dict(info=int(info),relative_linear_residual=float(error),calls=residual.evaluations-calls,
         construction_calls=construction_calls,seconds=time.perf_counter()-start,**diagnostics)
np.savez_compressed(ROOT/'linear_wide.npz',x=x,dx=dx)
(ROOT/'linear_wide.json').write_text(json.dumps(row,indent=2),encoding='utf8')
print(json.dumps(row),flush=True)
