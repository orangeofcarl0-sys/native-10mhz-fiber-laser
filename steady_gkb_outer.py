"""Pure adaptive GKB trust region in the existing scaled-state metric."""
import time
import numpy as np
from steady_gkb import gkb,svd_trust


def solve_gkb(residual,jtv,initial,max_steps=40,radius=.00625,root_threshold=1e-7,observer=None):
    x=initial.copy();r=residual(x);history=[];status='iteration_budget_reached'
    for outer in range(max_steps):
        norm=float(np.linalg.norm(r))
        if norm<root_threshold:status='residual_converged';break
        began=time.perf_counter()
        def derivative(v,h=1e-5):
            eps=h/max(np.linalg.norm(v),1e-100)
            return (residual(x+eps*v)-residual(x-eps*v))/(2*eps)
        u,v,jv,B,ladder=gkb(derivative,lambda w:jtv(x,w),-r,64,radius)
        target=np.zeros(len(B));target[0]=norm
        d=v[:,0];jd=jv[:,0];slope=float(r@jd);curvature=float(jd@jd)
        if slope>=0 or curvature<=0:
            status='no_verified_descent';break
        row=dict(step=outer,residual=norm,radius=radius,k=v.shape[1],
                 gkb_seconds=time.perf_counter()-began,ladder=ladder,
                 saturated=bool(ladder[-1].get('saturated',False)),
                 singular_values=np.linalg.svd(B,compute_uv=False).tolist(),
                 u_orth=float(np.linalg.norm(u.T@u-np.eye(u.shape[1]))),
                 v_orth=float(np.linalg.norm(v.T@v-np.eye(v.shape[1]))),
                 bidiagonal_relative=float(np.linalg.norm(jv-u@B)/max(np.linalg.norm(jv),1e-100)),trials=[])
        history.append(row)
        def trial(step,pred,kind,used_radius,pc):
            feasible=bool(residual.feasible(x+step));rr=residual(x+step) if feasible else None
            checks=[]
            for h in [1e-5,3e-6,1e-6]:
                js=derivative(step,h);p=float(-r@js-.5*(js@js))
                checks.append(dict(h=h,prediction=p,relative=abs(p-pred)/max(abs(pred),1e-100)))
            actual=float((r@r-rr@rr)/2) if feasible else None
            rho=actual/pred if feasible and pred>0 else -1.
            passed=bool(feasible and pred>0 and actual>0 and rho>.1
                        and pred>=pc*(1-1e-6) and np.linalg.norm(step)<=used_radius*(1+1e-6)
                        and all(c['prediction']>0 and c['relative']<.05 for c in checks)
                        and actual/checks[-1]['prediction']>.1)
            t=dict(kind=kind,radius=used_radius,prediction=float(pred),actual=actual,rho=rho,
                   residual=float(np.linalg.norm(rr)) if feasible else None,
                   step_norm=float(np.linalg.norm(step)),feasible=feasible,checks=checks,
                   cauchy_prediction=pc,passed=passed)
            row['trials'].append(t)
            return rr,t
        accepted=False
        for attempt in range(14):
            used_radius=radius
            y,pred,lam=svd_trust(B,target,radius);step=v@y
            alpha=min(radius,-slope/curvature);sc=alpha*d
            pc=float(-alpha*slope-.5*alpha**2*curvature)
            kind='gkb'
            if pred<pc*(1-1e-6) or np.linalg.norm(step)>radius*(1+1e-6):
                step=sc;pred=pc;kind='cauchy_model_fallback'
            rr,t=trial(step,pred,kind,used_radius,pc);t['lambda_value']=lam
            if t['passed']:
                accepted=True
                radius=used_radius*.25 if t['rho']<.25 else min(2.,2*used_radius) if t['rho']>.75 and t['step_norm']>.8*used_radius else used_radius
                break
            radius=used_radius*.25
        if not accepted:
            rr,t=trial(sc,pc,'cauchy_actual_fallback',used_radius,pc)
            accepted=t['passed']
            if accepted:step=sc;radius=used_radius
        if accepted:
            row.update(accepted_trial=len(row['trials'])-1,next_radius=radius)
            x=x+step;r=rr
        else:status='globalization_stalled'
        row['seconds']=time.perf_counter()-began
        if observer:observer(x,r,history)
        if not accepted:break
        if radius<1e-10:status='radius_floor_reached';break
    if np.linalg.norm(r)<root_threshold:status='residual_converged'
    return x,history,status
