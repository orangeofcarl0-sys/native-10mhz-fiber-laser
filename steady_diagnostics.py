"""State-scaled Newton diagnostics; no inference of physical stability."""
import numpy as np


def blocks(residual, step):
    if hasattr(residual,'step_blocks'):
        return residual.step_blocks(step)
    n=residual.n
    return dict(field=float(np.linalg.norm(step[:4*n])),
                population_rms=float(np.linalg.norm(step[4*n:-2])),
                phase_rad=float(abs(step[-2])),time_scaled=float(abs(step[-1])),
                shift_ps=float(abs(step[-1]*residual.time_scale)),
                total=float(np.linalg.norm(step)))


def gauge_geometry(residual,x):
    a,_,_,_=residual.unpack(x)
    a=a/residual.scale
    derivative=np.fft.ifft(1j*residual.omega*np.fft.fft(a))
    rows=[1j*residual.template,residual.time_tangent]
    columns=[1j*a,derivative]
    matrix=np.array([[np.vdot(u,v).real/max(np.linalg.norm(u)*np.linalg.norm(v),1e-100)
                      for v in columns] for u in rows])
    singular=np.linalg.svd(matrix,compute_uv=False)
    return dict(matrix=matrix.tolist(),condition=float(singular[0]/max(singular[-1],1e-100)),
                smallest_singular=float(singular[-1]))


def reanchor(residual,x):
    """Change only the gauge slice; retain field scale, time scale and drifts."""
    a,_,_,_=residual.unpack(x)
    template=a/residual.scale
    phase=1j*template
    tangent=np.fft.ifft(1j*residual.omega*np.fft.fft(template))
    tangent-=phase*np.vdot(phase,tangent).real/max(np.linalg.norm(phase)**2,1e-100)
    if np.linalg.norm(tangent)<1e-12:
        raise ValueError('Time gauge requires a nonconstant pulse')
    residual.template=template
    residual.time_tangent=tangent/np.linalg.norm(tangent)


def directional_tests(residual,x,r,direction):
    """Independent central Jd, epsilon sweep and forward/symmetric Taylor tests."""
    norm=np.linalg.norm(direction)
    unit=direction/norm
    central=(residual(x+1e-6*unit)-residual(x-1e-6*unit))/2e-6*norm
    denominator=max(np.linalg.norm(central),1e-100)
    epsilon_rows=[]
    for epsilon in [1e-5,1e-6,1e-7,1e-8]:
        plus,minus=residual(x+epsilon*unit),residual(x-epsilon*unit)
        forward=(plus-r)/epsilon*norm
        symmetric=(plus-minus)/(2*epsilon)*norm
        epsilon_rows.append(dict(epsilon=epsilon,
            forward_relative=float(np.linalg.norm(forward-central)/denominator),
            central_relative=float(np.linalg.norm(symmetric-central)/denominator)))
    rows=[]
    for h in 2.**(-np.arange(8)):
        trial=x+h*direction
        prediction=.5*(np.linalg.norm(r)**2-np.linalg.norm(r+h*central)**2)
        row=dict(h=float(h),feasible=bool(residual.feasible(trial)),predicted_reduction=float(prediction))
        if row['feasible']:
            rp=residual(trial)
            actual=.5*(np.linalg.norm(r)**2-np.linalg.norm(rp)**2)
            row.update(residual=float(np.linalg.norm(rp)),actual_reduction=float(actual),
                       rho=float(actual/prediction) if prediction>0 else None,
                       forward_taylor=float(np.linalg.norm(rp-r-h*central)/(h*denominator)))
            if residual.feasible(x-h*direction):
                rm=residual(x-h*direction)
                row['symmetric_taylor']=float(np.linalg.norm(rp-rm-2*h*central)/(2*h*denominator))
        rows.append(row)
    _,population,_,_=residual.unpack(x)
    return dict(step_blocks=blocks(residual,direction),gauge=gauge_geometry(residual,x),
                population_boundary_distance=float(min(population.min(),1-population.max())),
                central_linear_residual=float(np.linalg.norm(r+central)/np.linalg.norm(r)),
                epsilon_sweep=epsilon_rows,taylor=rows)
