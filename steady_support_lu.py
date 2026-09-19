"""Factor the same coarse Jacobian; verify solve residual, retain SVD fallback."""
import warnings
import numpy as np
from scipy.linalg import lu_factor,lu_solve,get_lapack_funcs,LinAlgWarning
from scipy.sparse.linalg import LinearOperator
from steady_support import SUPPORTS,residual_support
from steady_preconditioner import spectral_coordinates
from steady_window import center,embed

def build_factored(residual,x,r,label='A',difference='forward',epsilon=1e-6,audit=None,descent=None):
    if difference not in ('forward','central') or epsilon<=0:
        raise ValueError('Invalid coarse difference method/step')
    localized,cutoff=SUPPORTS[label];n=residual.n//2 if localized else residual.n
    restrict0,lift0,size=spectral_coordinates(n,residual.cells,cutoff)
    restrict=(lambda v:restrict0(center(v,n))) if localized else restrict0
    lift=(lambda v:embed(lift0(v),n)) if localized else lift0
    matrix=np.empty((size,size))
    full_gradient=np.zeros(size) if descent is not None else None
    if audit is not None:
        gradient=np.zeros(size);projected_gradient=np.zeros(size);forward_gradient=np.zeros(size);frobenius2=0.
    for start in range(0,size,16):
        count=min(16,size-start);basis=np.zeros((count,size));basis[np.arange(count),start+np.arange(count)]=1.
        perturbations=np.array([epsilon*lift(v) for v in basis])
        plus=residual.batch(x+perturbations)
        if difference=='central' or audit is not None:
            minus=residual.batch(x-perturbations)
            central=(plus-minus)/(2*epsilon)
        responses=central if difference=='central' else (plus-r)/epsilon
        for j,value in enumerate(responses):matrix[:,start+j]=restrict(value)
        if descent is not None:
            for j,value in enumerate(responses):full_gradient[start+j]=np.dot(value,r)
        if audit is not None:
            for j,value in enumerate(central):
                gradient[start+j]=np.dot(value,r)
                projected_gradient[start+j]=np.dot(restrict(value),restrict(r))
                forward_gradient[start+j]=np.dot((plus[j]-r)/epsilon,r)
                frobenius2+=np.dot(value,value)
    if audit is not None:
        audit.update(gradient=gradient,projected_gradient=projected_gradient,
                     forward_gradient=forward_gradient,frobenius_norm=float(np.sqrt(frobenius2)),
                     gradient_difference='central',gradient_epsilon=epsilon)
    if descent is not None:
        # Reuse full response columns before restriction; no extra map calls.
        descent.update(matrix=matrix,restrict=restrict,lift=lift,gradient=full_gradient)
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
        coarse_difference=difference,coarse_epsilon=epsilon,
        coarse_condition_1_estimate=float(1/max(rcond,1e-300)),factor_probe_residual=probe,
        **residual_support(residual,r,label))
