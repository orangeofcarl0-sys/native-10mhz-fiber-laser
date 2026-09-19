"""Approximate optical inverse and small inversion/gauge Schur complement.

Only the Newton linear solve is preconditioned. The nonlinear residual and
physical cavity parameters are unchanged. All coordinates use CavityResidual.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator


def optical_transfer(residual, x):
    """Frozen-population linear optical transfer, shape [frequency, 2, 2].

    Retains segment order, dispersion, Jones rotations, gain bandwidth and
    losses. Kerr and CNT differential response are omitted in this approximation.
    """
    e, c = residual.engine.e, residual.engine.c
    _, population, phase, shift = residual.unpack(x)
    transfer = np.broadcast_to(np.eye(2, dtype=complex), (residual.n, 2, 2)).copy()
    for index, (steps, dz, half, rotation, _) in enumerate(e.ops):
        diagonal = half ** (2 * steps)
        if index == 1:
            gain = c['emission_s_m'] * population - c['alpha_s_m'] * (1-population)
            diagonal *= np.exp(.5 * dz * gain.sum() * e.profile)
        segment = np.einsum('ij,jk,lj->kil', rotation, diagonal, rotation)
        transfer = segment @ transfer
        if index == 3:
            transfer = e.J @ transfer
            oc = float(residual.engine.oc[0])
            loss_db = c['oc_excess_dB'] + c['splice_connector_dB']
            transfer *= np.sqrt((1-oc)*(1-c['sa_modulation']-c['sa_nonsaturable']))
            transfer *= 10**(-loss_db/20)
    transfer *= 10**(-c['hybrid_IL_dB']/20)
    transfer[:, 1, :] *= 10**(-c['hybrid_PDL_dB']/20)
    transfer *= np.exp(1j*residual.omega*shift-1j*phase)[:, None, None]
    return transfer


def schur_inverse(field_inverse, upper, lower, diagonal):
    """Inverse of [A U; V D] using S=D-V A^-1 U; expose conditioning.

    A is represented by its inverse action; U,V,D are small-rank dense blocks.
    SVD cutoff handles singular approximate blocks without claiming exactness.
    """
    inverse_upper = np.column_stack([field_inverse(v) for v in upper.T])
    schur = diagonal - lower @ inverse_upper
    singular = np.linalg.svd(schur, compute_uv=False)
    schur_pinv = np.linalg.pinv(schur, rcond=1e-10)
    m = upper.shape[0]
    def apply(v):
        z = field_inverse(v[:m])
        tail = schur_pinv @ (v[m:] - lower @ z)
        return np.r_[z-inverse_upper @ tail, tail]
    condition = float(singular[0]/max(singular[-1], 1e-300))
    return LinearOperator((m+len(diagonal),)*2, matvec=apply, dtype=float), condition


def build(residual, x, r, damping=.03):
    """Rebuild once per Newton step; apply without additional cavity calls.

    Approximate lower optical-to-population coupling by its energy direction.
    All 15 population and 2 gauge columns are differentiated through the full map.
    Damping regularizes only the optical inverse, not the equation being solved.
    """
    m, n = 4*residual.n, residual.n
    a = (x[:2*n] + 1j*x[2*n:m]).reshape(2, n)
    b = a + (r[:2*n] + 1j*r[2*n:m]).reshape(2, n)
    transfer = optical_transfer(residual, x)
    predicted = np.fft.ifft(np.einsum('kij,jk->ik', transfer, np.fft.fft(a)))
    fit = np.vdot(predicted, b)/max(np.vdot(predicted, predicted).real, 1e-100)
    inverse = np.linalg.inv(fit*transfer-(1+damping)*np.eye(2))
    def field_inverse(v):
        z = (v[:2*n] + 1j*v[2*n:]).reshape(2, n)
        z = np.fft.ifft(np.einsum('kij,jk->ik', inverse, np.fft.fft(z)))
        return np.r_[z.real.ravel(), z.imag.ravel()]
    h = 1e-6
    columns = []
    for j in range(m, len(x)):
        trial = x.copy(); trial[j] += h
        columns.append((residual(trial)-r)/h)
    columns = np.column_stack(columns)
    direction = x[:m]/max(np.linalg.norm(x[:m]), 1e-100)
    trial = x.copy(); trial[:m] += h*direction
    response = (residual(trial)-r)/h
    lower = np.outer(response[m:], direction)
    for j, tangent in enumerate([1j*residual.template, residual.time_tangent]):
        lower[-2+j] = np.r_[tangent.real.ravel(), tangent.imag.ravel()]
    operator, condition = schur_inverse(field_inverse, columns[:m], lower, columns[m:])
    return operator, dict(schur_condition=condition, optical_fit_abs=float(abs(fit)),
                         optical_fit_phase=float(np.angle(fit)), damping=damping)


def spectral_coordinates(n, cells, cutoff, indices=None):
    """Orthonormal restriction/lifting for both complex polarizations and tail."""
    if not 0 < cutoff < n//2:
        raise ValueError('Cutoff must lie strictly inside the FFT grid')
    if indices is None:
        indices = np.r_[np.arange(cutoff+1), np.arange(n-cutoff,n)]
    indices = np.asarray(indices,dtype=int)
    if len(np.unique(indices)) != len(indices) or np.any((indices < 0) | (indices >= n)):
        raise ValueError('Fourier indices must be distinct and within the grid')
    size = 2*len(indices)
    def restrict(x):
        field = (x[:2*n]+1j*x[2*n:4*n]).reshape(2,n)
        coefficients = np.fft.fft(field,norm='ortho')[:,indices].ravel()
        return np.r_[coefficients.real,coefficients.imag,x[4*n:]]
    def lift(v):
        coefficients = np.zeros((2,n),complex)
        coefficients[:,indices] = (v[:size]+1j*v[size:2*size]).reshape(2,-1)
        field = np.fft.ifft(coefficients,norm='ortho')
        return np.r_[field.real.ravel(),field.imag.ravel(),v[2*size:]]
    return restrict, lift, 2*size+cells+2


def build_coarse(residual, x, r, cutoff=32, indices=None, audit=None):
    """Full real Jacobian in a Fourier subspace, identity-negated complement.

    All differentiated maps run on the ORIGINAL fine grid. Only the inverse is
    projected; the cavity, residual and Krylov Jv are not spectrally truncated.
    Construction cost must be counted, unlike a free approximate preconditioner.
    """
    restrict,lift,size = spectral_coordinates(residual.n,residual.cells,cutoff,indices)
    matrix = np.empty((size,size))
    for start in range(0,size,16):
        trials=[]
        for j in range(start,min(start+16,size)):
            basis=np.zeros(size);basis[j]=1
            trials.append(x+1e-6*lift(basis))
        responses = residual.batch(np.array(trials)) if hasattr(residual,'batch') else np.array([residual(v) for v in trials])
        for offset,response in enumerate(responses):
            matrix[:,start+offset]=restrict((response-r)/1e-6)
    u,s,vh = np.linalg.svd(matrix,full_matrices=False)
    if audit is not None:
        audit(matrix,restrict(r),s,vh)
    keep = s > s[0]*1e-10
    inverse = (vh[keep].T/s[keep])@u[:,keep].T
    def apply(v):
        coarse = restrict(v)
        return lift(inverse@coarse) - (v-lift(coarse))
    projection_error = np.linalg.norm(r-lift(restrict(r)))/np.linalg.norm(r)
    return LinearOperator((len(x),len(x)),matvec=apply,dtype=float), dict(
        coarse_dimension=size,cutoff=cutoff if indices is None else None,coarse_rank=int(keep.sum()),
        coarse_condition=float(s[0]/max(s[-1],1e-300)),
        residual_outside_subspace=float(projection_error))
