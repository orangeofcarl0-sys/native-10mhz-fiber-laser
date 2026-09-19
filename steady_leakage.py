"""Audit full responses of weak projected modes; alter only inverse action."""
import numpy as np
from scipy.sparse.linalg import LinearOperator
from steady_preconditioner import spectral_coordinates
from steady_window import center,embed


def geometry(residual,x,r,cutoff=384):
    n=residual.n//2
    restrict0,lift0,size=spectral_coordinates(n,residual.cells,cutoff)
    restrict=lambda v:restrict0(center(v,n))
    lift=lambda v:embed(lift0(v),n)
    matrix=np.empty((size,size))
    for start in range(0,size,16):
        basis=np.eye(size)[start:start+16]
        responses=residual.batch(np.array([x+1e-6*lift(v) for v in basis]))
        for k,response in enumerate(responses):matrix[:,start+k]=restrict((response-r)/1e-6)
    u,s,vh=np.linalg.svd(matrix,full_matrices=False)
    modes=[]
    for i in range(max(0,size-20),size):
        v=lift(vh[i]);d=(residual(x+1e-6*v)-residual(x-1e-6*v))/2e-6
        projected=restrict(d);outside=d-lift(projected)
        modes.append(dict(index=int(i),sigma=float(s[i]),full_norm=float(np.linalg.norm(d)),
            leakage=float(np.linalg.norm(outside)/max(np.linalg.norm(projected),1e-100))))
    return restrict,lift,u,s,vh,modes


def inverse(data,length,mode='baseline',threshold=10.):
    restrict,lift,u,s,vh,modes=data
    weights=np.divide(1.,s,out=np.zeros_like(s),where=s>s[0]*1e-10)
    bad=[v for v in modes if v['leakage']>threshold]
    for v in bad:
        if mode=='truncate':weights[v['index']]=0.
        elif mode=='cap':weights[v['index']]=1/max(v['sigma'],v['full_norm'])
    dense=(vh.T*weights)@u.T
    def apply(v):
        c=restrict(v)
        return lift(dense@c)-(v-lift(c))
    return LinearOperator((length,length),matvec=apply,dtype=float),dict(
        leakage_mode=mode,leakage_threshold=threshold,leakage_modes=modes,
        modified_modes=len(bad) if mode!='baseline' else 0,
        coarse_dimension=len(s),coarse_condition=float(s[0]/s[-1]))


def build_leakage(residual,x,r,cutoff=384,mode='cap',threshold=10.):
    return inverse(geometry(residual,x,r,cutoff),len(x),mode,threshold)
