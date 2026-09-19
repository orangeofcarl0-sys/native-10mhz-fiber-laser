"""Current endpoint: projected SVD, full-objective subspace gradient and knobs."""
import json,time
import numpy as np
from config import ROOT,PROJECT
from steady_parameters import parameter_residual,parameter_column,PARAMETERS
from steady_window import center,embed
from steady_preconditioner import spectral_coordinates

state=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
residual=parameter_residual(state,'pump',gpu=True);x=state['x'];r=residual(x)
n=residual.n//2;restrict,lift,size=spectral_coordinates(n,residual.cells,384)
project=lambda v:restrict(center(v,n))
expand=lambda v:embed(lift(v),n)
matrix=np.empty((size,size));gfull=np.empty(size);column_norms=np.empty(size)
start=time.perf_counter()
for first in range(0,size,16):
    probes=[]
    for j in range(first,min(first+16,size)):
        e=np.zeros(size);e[j]=1;probes.append(x+1e-6*expand(e))
    derivatives=(residual.batch(np.array(probes))-r)/1e-6
    for offset,d in enumerate(derivatives):
        j=first+offset;matrix[:,j]=project(d);gfull[j]=np.dot(d,r);column_norms[j]=np.dot(d,d)
u,s,vh=np.linalg.svd(matrix,full_matrices=False);rc=project(r);coeff=u.T@rc
gc=matrix.T@rc;direction=-expand(gfull)/np.linalg.norm(gfull)
jd=(residual(x+1e-6*direction)-residual(x-1e-6*direction))/(2e-6)
decreases=[]
for h in [1e-6,1e-5,1e-4,1e-3,.01]:
    rr=residual(x+h*direction)
    decreases.append(dict(step=h,residual=float(np.linalg.norm(rr))))
modes=[dict(index=j,sigma=float(s[-j]),left_residual=float(coeff[-j]),
            newton_coefficient=float(-coeff[-j]/s[-j])) for j in range(1,21)]
result=dict(residual=float(np.linalg.norm(r)),field_block_fraction=float(np.dot(r[:4*residual.n],r[:4*residual.n])/np.dot(r,r)),
    projected_gradient_norm=float(np.linalg.norm(gc)),normalized_projected_gradient=float(np.linalg.norm(gc)/(s[0]*np.linalg.norm(rc))),
    full_objective_subspace_gradient_norm=float(np.linalg.norm(gfull)),
    full_gradient_directional_derivative=float(np.dot(r,jd)),
    full_gradient_directional_prediction=float(-np.linalg.norm(gfull)),
    full_columns_frobenius_norm=float(np.sqrt(column_norms.sum())),gradient_trials=decreases,
    sigma_min=float(s[-1]),condition=float(s[0]/s[-1]),modes=modes,
    soft20_newton_norm_fraction=float(np.linalg.norm(coeff[-20:]/s[-20:])/np.linalg.norm(coeff/s)),parameters=[])
np.savez_compressed(ROOT/'parameter_geometry_cache.npz',matrix=matrix,u=u,s=s,vh=vh,gfull=gfull)
np.savez_compressed(ROOT/'parameter_geometry_modes.npz',left=u[:,-20:],right=vh[-20:],singular=s[-20:],coefficients=coeff[-20:],gradient=gfull)
print('GRADIENT',json.dumps({k:v for k,v in result.items() if k not in ('modes','parameters')}),flush=True)
v=vh[-1];target=np.r_[-rc,0.]
for name,(base,scale,unit) in PARAMETERS.items():
    column=parameter_column(state,name,x,h=1e-3,gpu=True)
    refined=parameter_column(state,name,x,h=5e-4,gpu=True)
    b=project(refined);coupling=float(np.dot(u[:,-1],b))
    bordered=np.zeros((size+1,size+1));bordered[:size,:size]=matrix
    bordered[:size,-1]=b;bordered[-1,:size]=v
    sb=np.linalg.svd(bordered,compute_uv=False);answer=np.linalg.solve(bordered,target)
    row=dict(name=name,base=base,scale=scale,unit=unit,
        derivative_norm_per_scaled_parameter=float(np.linalg.norm(refined)),
        derivative_norm_per_physical_unit=float(np.linalg.norm(refined)/scale),
        halving_relative_difference=float(np.linalg.norm(column-refined)/max(np.linalg.norm(refined),1e-100)),
        left_soft_coupling=coupling,eta_projected=float(abs(coupling)/max(np.linalg.norm(b),1e-100)),
        eta_full_denominator=float(abs(coupling)/max(np.linalg.norm(refined),1e-100)),
        soft20_coupling=np.abs(u[:,-20:].T@b).tolist(),bordered_sigma_min=float(sb[-1]),
        bordered_condition=float(sb[0]/sb[-1]),linear_parameter_change=float(answer[-1]),
        linear_state_step_norm=float(np.linalg.norm(answer[:-1])))
    result['parameters'].append(row)
    np.savez_compressed(ROOT/(name+'_parameter_column.npz'),full=refined)
    print('PARAMETER',json.dumps(row),flush=True)
    result['seconds']=time.perf_counter()-start
    (ROOT/'parameter_geometry.json').write_text(json.dumps(result,indent=2),encoding='utf8')
