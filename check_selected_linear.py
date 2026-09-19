"""Actual spectral support, complete fine-grid Jacobian columns, right inverse."""
import os,json,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator,lgmres
from config import ROOT,config
from steady_state import CavityResidual
from steady_preconditioner import build_coarse

saved=np.load(Path(os.environ['LASER_STEADY_DIR'])/'best_pilot_larger_krylov.npz')
residual=CavityResidual(config('CNT_OC',.2),.125,saved['a'],.05,.8,gpu=True)
support=np.load(ROOT/'spectral_support.npz');x=support['x'];r=residual(x)
np.testing.assert_allclose(r,support['r'],atol=1e-12)
# Check the actual nonlinear state, not only a Gaussian CPU test.
probes=np.repeat(x[None],3,axis=0)
rng=np.random.default_rng(3);probes+=rng.normal(size=probes.shape)*1e-8
reference=np.array([residual(v) for v in probes]);batch=residual.batch(probes)
batch_error=np.linalg.norm(reference-batch)/np.linalg.norm(reference)
assert batch_error < 1e-10
def jv(v):
    h=1e-7/max(np.linalg.norm(v),1e-100)
    return (residual(x+h*v)-r)/h
jacobian=LinearOperator((len(x),len(x)),matvec=jv,dtype=float)
rows=[]
for count in [257,513]:
    start=time.perf_counter();calls=residual.evaluations
    preconditioner,diagnostics=build_coarse(residual,x,r,indices=support['order'][:count])
    construction_calls=residual.evaluations-calls
    operator=LinearOperator(jacobian.shape,matvec=lambda v:jacobian@(preconditioner@v),dtype=float)
    d,info=lgmres(operator,-r,rtol=.03,atol=0,maxiter=2,inner_m=60,outer_k=3)
    dx=preconditioner@d
    error=np.linalg.norm(jv(dx)+r)/np.linalg.norm(r)
    row=dict(selected_bins=count,info=int(info),relative_linear_residual=float(error),
             calls=residual.evaluations-calls,construction_calls=construction_calls,
             seconds=time.perf_counter()-start,batch_error=float(batch_error),**diagnostics)
    rows.append(row)
    np.savez_compressed(ROOT/f'linear_selected_{count}.npz',x=x,dx=dx)
    (ROOT/'linear_selected.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    print(json.dumps(row),flush=True)
