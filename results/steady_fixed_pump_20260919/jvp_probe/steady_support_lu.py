"""Factor the same coarse Jacobian; verify solve residual, retain SVD fallback."""
import warnings
import numpy as np
from scipy.linalg import lu_factor,lu_solve,get_lapack_funcs,LinAlgWarning
from scipy.sparse.linalg import LinearOperator
from steady_support import SUPPORTS,residual_support
from steady_preconditioner import spectral_coordinates
from steady_window import center,embed

def build_factored(residual,x,r,label='A'):
    localized,cutoff=SUPPORTS[label];n=residual.n//2 if localized else residual.n
    restrict0,lift0,size=spectral_coordinates(n,residual.cells,cutoff)
    restrict=(lambda v:restrict0(center(v,n))) if localized else restrict0
    lift=(lambda v:embed(lift0(v),n)) if localized else lift0
    matrix=np.empty((size,size))
    for start in range(0,size,16):
        count=min(16,size-start);basis=np.zeros((count,size));basis[np.arange(count),start+np.arange(count)]=1.
        response=residual.batch(np.array([x+1e-6*lift(v) for v in basis]))
        for j,value in enumerate(response):matrix[:,start+j]=restrict((value-r)/1e-6)
    norm1=np.linalg.norm(matrix,1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',LinAlgWarning)
            lu,piv=lu_factor(matrix)
        rcond,flag=get_lapack_funcs('gecon',(lu,))(lu,norm1)
    except LinAlgWarning:
        rcond,flag=0.,0
    if flag!=0:raise RuntimeError('Coarse condition estimation failed')
    mode='LU'
    if rcond<1e-9:
        u,s,vh=np.linalg.svd(matrix,full_matrices=False);keep=s>s[0]*1e-10
        dense=(vh[keep].T/s[keep])@u[:,keep].T
        solve=lambda rhs:dense@rhs;mode='SVD fallback'
    else:solve=lambda rhs:lu_solve((lu,piv),rhs)
    rhs=np.random.default_rng(42).normal(size=size);solution=solve(rhs)
    probe=float(np.linalg.norm(matrix@solution-rhs)/np.linalg.norm(rhs))
    if mode=='LU' and probe>1e-7:raise RuntimeError('Coarse LU solve failed backward check')
    def apply(v):
        c=restrict(v);return lift(solve(c))-(v-lift(c))
    return LinearOperator((len(x),len(x)),matvec=apply,dtype=float),dict(
        support_label=label,coarse_dimension=size,factorization=mode,
        coarse_condition_1_estimate=float(1/max(rcond,1e-300)),factor_probe_residual=probe,
        **residual_support(residual,r,label))
