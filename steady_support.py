"""Orthogonal temporal/spectral residual accounting and support controls."""
import numpy as np
from steady_preconditioner import spectral_coordinates,build_coarse
from steady_window import center,embed,build_localized

SUPPORTS={'A':(True,384),'B':(True,768),'C':(False,768)}

def residual_support(residual,r,label):
    localized,cutoff=SUPPORTS[label];n=residual.n//2 if localized else residual.n
    restrict,lift,_=spectral_coordinates(n,residual.cells,cutoff)
    rt=center(r,n) if localized else r
    temporal=r-embed(rt,n) if localized else np.zeros_like(r)
    spectral_local=rt-lift(restrict(rt))
    spectral=embed(spectral_local,n) if localized else spectral_local
    coarse=r-temporal-spectral
    norm=np.linalg.norm(r)
    return dict(temporal_relative=float(np.linalg.norm(temporal)/norm),
        spectral_relative=float(np.linalg.norm(spectral)/norm),
        coarse_relative=float(np.linalg.norm(coarse)/norm),
        orthogonality=float(np.dot(temporal,spectral)/norm**2),
        squared_partition=float((np.dot(temporal,temporal)+np.dot(spectral,spectral)+np.dot(coarse,coarse))/norm**2),
        support_ps=n*residual.dt,bandwidth_GHz=1000*cutoff/(n*residual.dt))

def build_support(residual,x,r,label='A'):
    localized,cutoff=SUPPORTS[label]
    op,info=(build_localized if localized else build_coarse)(residual,x,r,cutoff=cutoff)
    info.update(support_label=label,**residual_support(residual,r,label))
    return op,info
