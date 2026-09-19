"""Three fixed-state tests; no state advancement and no energy merit row."""
import json,time,hashlib
import numpy as np
from config import ROOT,PROJECT
from steady_parameters import parameter_residual
from steady_preconditioner import spectral_coordinates
from steady_support_lu import build_factored
from steady_hookstep import arnoldi,hookstep

prior=PROJECT/'results/steady_fixed_pump_20260919'
template=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
sources=['run_merit_geometry.py','steady_support_lu.py','steady_support.py','steady_hookstep.py','steady_state.py','steady_parameters.py']
hashes={p:hashlib.sha256((PROJECT/p).read_bytes()).hexdigest() for p in sources}
radii=[.00156,.003125,.00625,.0125,.025,.05]

def directional(residual,x,v,epsilon=1e-5):
    h=epsilon/max(np.linalg.norm(v),1e-100)
    return (residual(x+h*v)-residual(x-h*v))/(2*h)

def linear_control(residual,x,r,pre):
    h,z,y,predicted=arnoldi(lambda v:directional(residual,x,v),r,pre,limit=240,tolerance=.008)
    d=z@y
    checks=[dict(epsilon=eps,relative=float(np.linalg.norm(r+directional(residual,x,d,eps))/np.linalg.norm(r))) for eps in [1e-4,1e-5,1e-6]]
    return h,z,d,dict(krylov=len(y),predicted_relative=predicted,newton_norm=float(np.linalg.norm(d)),checks=checks)

def trial(residual,x,r,step,predicted):
    feasible=bool(residual.feasible(x+step));rr=residual(x+step) if feasible else None
    actual=float(.5*(np.dot(r,r)-np.dot(rr,rr))) if feasible else None
    return dict(feasible=feasible,residual=float(np.linalg.norm(rr)) if feasible else None,
        predicted_reduction=float(predicted),actual_reduction=actual,
        rho=actual/predicted if feasible and predicted>0 else None,
        residual_decrease_fraction=1-float(np.linalg.norm(rr)/np.linalg.norm(r)) if feasible else None)

for name in ['down_270','down_275','up_2775']:
    metadata=json.loads((prior/(name+'.json')).read_text());x=np.load(prior/(name+'.npz'))['x']
    residual=parameter_residual(template,'pump',(metadata['pump_W']-.05)/.005,gpu=True);r=residual(x)
    base=dict(name=name,pump_W=metadata['pump_W'],physical_residual=float(np.linalg.norm(r)),
        input=str((prior/(name+'.npz')).relative_to(PROJECT)).replace('\\','/'),
        input_sha256=hashlib.sha256((prior/(name+'.npz')).read_bytes()).hexdigest(),code_sha256=hashes)
    print('BUILD',name,'forward',flush=True);audit={} if name!='up_2775' else None
    start=time.perf_counter();pre,info=build_factored(residual,x,r,'C',audit=audit)
    base.update(build_seconds=time.perf_counter()-start,coarse=info)
    h,z,d,linear=linear_control(residual,x,r,pre);base['linear']=linear
    arrays=dict(newton_direction=d,residual=r)
    if audit is not None:
        _,lift,_=spectral_coordinates(residual.n,residual.cells,768)
        g=audit['gradient'];gnorm=np.linalg.norm(g);direction=-lift(g/gnorm)
        probes=[]
        for eps in [1e-4,1e-5,1e-6]:
            jd=directional(residual,x,direction,eps);derivative=float(np.dot(r,jd))
            rp=residual(x+eps*direction);rm=residual(x-eps*direction)
            merit_derivative=float((np.dot(rp,rp)-np.dot(rm,rm))/(4*eps))
            probes.append(dict(epsilon=eps,derivative=derivative,merit_derivative=merit_derivative,
                relative_gradient_disagreement=float(abs(derivative+gnorm)/gnorm)))
        jd=directional(residual,x,direction,1e-5);cauchy=float(gnorm/np.dot(jd,jd))
        gradient=dict(norm=float(gnorm),normalized_frobenius=float(gnorm/(audit['frobenius_norm']*np.linalg.norm(r))),
            frobenius_norm=audit['frobenius_norm'],dimension=len(g),gradient_epsilon=audit['gradient_epsilon'],
            projected_output_relative_error=float(np.linalg.norm(g-audit['projected_gradient'])/gnorm),
            forward_gradient_relative_error=float(np.linalg.norm(g-audit['forward_gradient'])/gnorm),
            response_cosine=float(gnorm/(np.linalg.norm(jd)*np.linalg.norm(r))),cauchy_step=cauchy,
            direction_norm=float(np.linalg.norm(direction)),probes=probes)
        line=[]
        for alpha in sorted(set(np.r_[np.geomspace(1e-8,.05,23),cauchy])):
            pred=float(.5*(np.dot(r,r)-np.linalg.norm(r+alpha*jd)**2))
            line.append(dict(step=float(alpha),**trial(residual,x,r,alpha*direction,pred)))
        sweep=[];steps=[]
        for radius in radii:
            step,pred,lam=hookstep(h,z,np.linalg.norm(r),radius,np.ones_like(x))
            jstep=directional(residual,x,step,1e-6)
            independent=float(.5*(np.dot(r,r)-np.linalg.norm(r+jstep)**2))
            row=dict(radius=radius,step_norm=float(np.linalg.norm(step)),lambda_value=lam,
                independent_predicted_reduction=independent,**trial(residual,x,r,step,pred))
            row['independent_rho']=row['actual_reduction']/independent if independent>0 and row['feasible'] else None
            sweep.append(row);steps.append(step)
        base.update(gradient=gradient,gradient_line=line,radius_sweep=sweep)
        arrays.update(gradient=g,projected_gradient=audit['projected_gradient'],forward_gradient=audit['forward_gradient'],
                      gradient_direction=direction,hooksteps=np.array(steps))
    else:
        base['forward_control']=dict(coarse=info,linear=linear,build_seconds=base['build_seconds'])
        (ROOT/(name+'_forward.json')).write_text(json.dumps(base,indent=2))
        print('BUILD',name,'central',flush=True);start=time.perf_counter()
        pre,info=build_factored(residual,x,r,'C',difference='central',epsilon=1e-6)
        elapsed=time.perf_counter()-start
        _,_,dc,lc=linear_control(residual,x,r,pre)
        base['central_control']=dict(coarse=info,linear=lc,build_seconds=elapsed)
        arrays['central_newton_direction']=dc
    base['evaluations']=residual.evaluations
    np.savez_compressed(ROOT/(name+'_directions.npz'),**arrays)
    (ROOT/(name+'.json')).write_text(json.dumps(base,indent=2))
    print('RESULT',name,json.dumps({k:v for k,v in base.items() if k not in ['gradient_line','radius_sweep','code_sha256']}),flush=True)
