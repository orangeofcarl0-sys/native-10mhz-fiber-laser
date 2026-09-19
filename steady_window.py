"""Zero embedding preserves scaled field norm, population, phase and time units."""
import numpy as np
from scipy.sparse.linalg import LinearOperator
from steady_state import CavityResidual
from steady_preconditioner import build_coarse


def embed(x,n):
    z=(x[:2*n]+1j*x[2*n:4*n]).reshape(2,n)
    z=np.pad(z,((0,0),(n//2,n//2)))
    return np.r_[z.real.ravel(),z.imag.ravel(),x[4*n:]]


def center(x,n):
    z=(x[:4*n]+1j*x[4*n:8*n]).reshape(2,2*n)[:,n//2:3*n//2]
    return np.r_[z.real.ravel(),z.imag.ravel(),x[8*n:]]


def restore(c,state,wide=False,gpu=False):
    pad=lambda a:np.pad(a,((0,0),(a.shape[-1]//2,)*2)) if wide else a
    residual=CavityResidual(c,float(state['dt']),pad(state['original_template']),.05,.8,gpu=gpu)
    residual.scale=float(state['scale']);residual.time_scale=float(state['time_scale'])
    residual.template=pad(state['gauge_template']);residual.time_tangent=pad(state['time_tangent'])
    return residual,embed(state['x'],state['a'].shape[-1]) if wide else state['x'].copy()


class EmbeddedResidual:
    """P Rwide(E x) on the SAME localized perturbation subspace as the old grid.

    This is a diagnostic, not a wide-window solver. Outer residual rows and new
    wide-window perturbations are excluded; full wide singular values are unknown.
    """
    def __init__(self,wide,n,base=None):
        self.wide,self.n,self.cells=wide,n,wide.cells
        self.outside=0 if base is None else base-embed(center(base,n),n)

    def __call__(self,x):return center(self.wide(self.outside+embed(x,self.n)),self.n)

    def batch(self,states):
        return np.array([center(r,self.n) for r in
                         self.wide.batch(np.array([self.outside+embed(x,self.n) for x in states]))])


def build_localized(wide,x,r,cutoff=384):
    """Local fine-grid inverse; -I on outer field. Full wide Krylov stays intact."""
    n=wide.n//2
    local=EmbeddedResidual(wide,n,base=x)
    inverse,info=build_coarse(local,center(x,n),center(r,n),cutoff=cutoff)
    def apply(v):
        vcenter=center(v,n)
        return embed(inverse@vcenter,n)-(v-embed(vcenter,n))
    info['preconditioner_scope']='center-window Fourier subspace; outer field -I'
    info['center_window_ps']=n*wide.dt
    return LinearOperator((len(x),len(x)),matvec=apply,dtype=float),info
