"""Orthogonal union of state subspaces, preserving the physical trust metric."""
import numpy as np


def orthogonal_union(basis,responses,cutoff=1e-12):
    """S=Q R; SVD R supplies an isometric retained state basis and its J images."""
    q,r=np.linalg.qr(basis,mode='reduced')
    u,s,vt=np.linalg.svd(r,full_matrices=False)
    keep=s>s[0]*cutoff
    transform=vt[keep].T/s[keep]
    return q@u[:,keep],responses@transform,dict(rank=int(keep.sum()),
        singular_values=s.tolist(),relative_cutoff=cutoff),transform
