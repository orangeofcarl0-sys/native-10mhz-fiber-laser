"""Taylor, gauge, window and projected-stationarity checks before hookstep."""
import json
import numpy as np
from config import ROOT,PROJECT,config
from steady_state import CavityResidual
from steady_diagnostics import directional_tests
from steady_preconditioner import build_coarse

rows=[];c=config('CNT_OC',.2)
for step in range(8,12):
    state=np.load(ROOT/f'original_step_{step}.npz')
    residual=CavityResidual(c,.125,state['template'],.05,.8,gpu=True)
    x,r,d=state['x'],state['r'],state['dx']
    np.testing.assert_allclose(residual(x),r,atol=1e-12)
    row=directional_tests(residual,x,r,d);row['step']=step
    _,p,_,_=residual.unpack(x);dp=d[4*residual.n:-2]*np.sqrt(residual.cells)
    bounds=np.r_[(1-p[dp>0])/dp[dp>0],-p[dp<0]/dp[dp<0]]
    row['population_feasible_alpha_limit']=float(bounds.min())
    rows.append(row)
    print(json.dumps(dict(step=step,gauge_condition=row['gauge']['condition'],
        central_linear_residual=row['central_linear_residual'],
        epsilon_1e7_error=row['epsilon_sweep'][2]['forward_relative'],
        full_step_rho=row['taylor'][0].get('rho'))),flush=True)
    (ROOT/'taylor_diagnostics.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
assert max(row['epsilon_sweep'][2]['forward_relative'] for row in rows)<1e-3

final=np.load(PROJECT/'results/steady_block_20260919/wide_newton.npz')
original=CavityResidual(c,.125,final['template'],.05,.8,gpu=True)
x=final['x'];r=original(x);a,p,phase,shift=original.unpack(x);n=original.n
pad=lambda v:np.pad(v,((0,0),(n//2,n//2)))
wide=CavityResidual(c,.125,pad(final['template']),.05,.8,gpu=True)
wide.time_scale=original.time_scale;wide.time_tangent=pad(original.time_tangent)
rw=wide(wide.pack(pad(a),p,phase,shift))
rf=pad((r[:2*n]+1j*r[2*n:4*n]).reshape(2,n))
rp=np.r_[rf.real.ravel(),rf.imag.ravel(),r[4*n:]]
window=dict(original_residual=float(np.linalg.norm(r)),wide_residual=float(np.linalg.norm(rw)),
    relative_norm_change=float(abs(np.linalg.norm(rw)/np.linalg.norm(r)-1)),
    residual_vector_relative_change=float(np.linalg.norm(rw-rp)/np.linalg.norm(r)),
    original_window_ps=1024,wide_window_ps=2048,dt_ps=.125)
(ROOT/'window_diagnostics.json').write_text(json.dumps(window,indent=2),encoding='utf8')
print('WINDOW',json.dumps(window),flush=True)

state=np.load(ROOT/'original_step_11.npz')
residual=CavityResidual(c,.125,state['template'],.05,.8,gpu=True)
x,r=state['x'],state['r']
def audit(matrix,rc,s,vh):
    count=769;field_size=4*count
    indices=np.r_[np.arange(385),np.arange(residual.n-384,residual.n)]
    bins=np.fft.fftfreq(residual.n)*residual.n
    soft=[]
    for j in range(1,5):
        v=vh[-j]
        z=(v[:2*count]+1j*v[2*count:field_size]).reshape(2,count)
        soft.append(dict(singular_value=float(s[-j]),
            field_fraction=float(np.sum(v[:field_size]**2)),
            population_fraction=float(np.sum(v[field_size:-2]**2)),
            phase_fraction=float(v[-2]**2),time_fraction=float(v[-1]**2),
            field_wing_fraction=float(np.sum(abs(z[:,abs(bins[indices])>64])**2))))
    gradient=matrix.T@rc
    result=dict(projected_gradient_norm=float(np.linalg.norm(gradient)),
        normalized_projected_gradient=float(np.linalg.norm(gradient)/(s[0]*np.linalg.norm(rc))),
        projected_residual=float(np.linalg.norm(rc)),sigma_max=float(s[0]),
        sigma_min=float(s[-1]),soft_modes=soft,
        limitation='Projected gradient and algebraic singular modes do not prove a full-space local minimum, absence of roots, or a physical bifurcation.')
    (ROOT/'coarse_stationarity.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    np.savez_compressed(ROOT/'coarse_soft_modes.npz',vectors=vh[-4:],singular_values=s[-4:],bins=bins[indices])
    print('COARSE',json.dumps(result),flush=True)
build_coarse(residual,x,r,cutoff=384,audit=audit)
