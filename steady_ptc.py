"""Experimental pseudo-time shift for R=(Phi(A)-A, N-Neq, gauges).

Field and population signs differ. Gauge equations stay algebraic. This is an
artificial solver flow, NOT the physical EDF transient or a stability certificate.
"""
import numpy as np
from scipy.linalg import lu_factor,lu_solve
from scipy.sparse.linalg import LinearOperator
from steady_window import center,embed,EmbeddedResidual
from steady_preconditioner import build_coarse,spectral_coordinates


def mass_sign(n,cells):
    return np.r_[-np.ones(4*n),np.ones(cells),np.zeros(2)]


def shifted_local_inverse(matrix,n,cells,mu,cutoff=384):
    restrict,lift,size=spectral_coordinates(n,cells,cutoff)
    sign=np.r_[-np.ones(size-cells-2),np.ones(cells),np.zeros(2)]
    factor=lu_factor(matrix+mu*np.diag(sign))
    def apply(v):
        local=center(v,n);small=restrict(local)
        projected=embed(lift(small),n)
        return embed(lift(lu_solve(factor,small)),n)-(v-projected)/(1+mu)
    size_full=8*n+cells+2
    return LinearOperator((size_full,size_full),matvec=apply,dtype=float)


def local_matrix(residual,x,r):
    n=residual.n//2;saved=[]
    def audit(matrix,*_):saved.append(matrix.copy())
    build_coarse(EmbeddedResidual(residual,n,base=x),center(x,n),center(r,n),cutoff=384,audit=audit)
    return saved[0]
