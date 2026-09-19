"""Right-preconditioned Newton–GMRES with a physical-coordinate hookstep.

    min ||beta e1-H y|| subject to ||D Z y|| <= radius, Z=M V.
    D acts on already scaled state coordinates; it is not a coefficient bound.
"""
import numpy as np
from scipy.optimize import brentq
from steady_diagnostics import blocks,gauge_geometry,reanchor
from steady_preconditioner import build_coarse


def arnoldi(jv,r,preconditioner,limit=120,tolerance=.03):
    beta=np.linalg.norm(r)
    v=np.zeros((len(r),limit+1));v[:,0]=-r/beta
    z=np.zeros((len(r),limit));h=np.zeros((limit+1,limit))
    for j in range(limit):
        z[:,j]=preconditioner@v[:,j]
        w=jv(z[:,j])
        for _ in range(2):
            projection=v[:,:j+1].T@w
            h[:j+1,j]+=projection
            w-=v[:,:j+1]@projection
        h[j+1,j]=np.linalg.norm(w)
        if h[j+1,j]>1e-13:
            v[:,j+1]=w/h[j+1,j]
        target=np.zeros(j+2);target[0]=beta
        y=np.linalg.lstsq(h[:j+2,:j+1],target,rcond=1e-12)[0]
        relative=np.linalg.norm(target-h[:j+2,:j+1]@y)/beta
        if relative<=tolerance or h[j+1,j]<=1e-13:
            break
    return h[:j+2,:j+1],z[:,:j+1],y,float(relative)


def hookstep(h,z,beta,radius,weights):
    """Whiten the ACTUAL step metric, then solve a small constrained LS problem."""
    if radius<=0 or np.any(weights<=0):
        raise ValueError('Radius and state metric weights must be positive')
    _,s,vh=np.linalg.svd(weights[:,None]*z,full_matrices=False)
    keep=s>s[0]*1e-12
    transform=vh[keep].T/s[keep]
    reduced=h@transform
    u,sigma,vh_small=np.linalg.svd(reduced,full_matrices=False)
    target=np.zeros(h.shape[0]);target[0]=beta
    rhs=u.T@target
    def coefficients(lam):
        return np.divide(sigma*rhs,sigma*sigma+lam,
                         out=np.zeros_like(sigma),where=sigma*sigma+lam>1e-30)
    lam=0.
    if np.linalg.norm(coefficients(0))>radius:
        high=max(float(np.max(sigma*sigma)),1.)
        while np.linalg.norm(coefficients(high))>radius:
            high*=4
        lam=brentq(lambda value:np.linalg.norm(coefficients(value))-radius,0,high,
                   xtol=1e-14,rtol=1e-12)
    y=transform@(vh_small.T@coefficients(lam))
    step=z@y
    model=target-h@y
    return step,float(.5*(beta*beta-np.dot(model,model))),float(lam)


def progress_window(history,window=5):
    """Operational stop diagnostic, not proof of a flat valley or missing root."""
    if len(history)<=window:return None
    gain=1-history[-1]['residual']/history[-1-window]['residual']
    rhos=[h['trials'][h['accepted_trial']]['rho'] for h in history[-1-window:-1]
          if 'accepted_trial' in h]
    good=sum(v>.5 for v in rhos)
    return dict(relative_decrease=float(gain),good_rho_count=good,
                label='progress' if gain>.1 else 'slow' if gain>=.02 else 'stagnating',
                stop=bool(gain<.02 and good>=4))


def solve_globalized(residual,x,mode='hookstep',max_steps=12,cutoff=384,
                     radius=.1,rebuild_every=3,reanchor_threshold=None,progress=None,
                     stop_window=None,preconditioner_builder=None,observer=None,
                     linear_limit=120,linear_tolerance=.03,linear_refresh_threshold=None):
    """Shared Arnoldi/preconditioner control; mode changes only globalization.

    Field normalized by initial template norm; population uses RMS units;
    phase uses rad, time uses the fixed initial derivative scale. Thus D=I.
    """
    if mode not in ('hookstep','line_search'):
        raise ValueError('Unknown globalization method')
    x=x.copy();r=residual(x);history=[];weights=np.ones_like(x)
    preconditioner=None;force_rebuild=False
    for iteration in range(max_steps+1):
        norm=np.linalg.norm(r)
        row=dict(step=iteration,residual=float(norm),calls=residual.evaluations,
                 gauge=gauge_geometry(residual,x),radius=float(radius))
        history.append(row)
        if stop_window is not None:
            row['progress_window']=progress_window(history,stop_window)
        if progress:progress(row)
        if norm<1e-7:return x,history,'residual_converged'
        if row.get('progress_window') and row['progress_window']['stop']:
            return x,history,'progress_stagnated'
        if iteration==max_steps:break
        rebuild=force_rebuild or iteration%rebuild_every==0 or (
            iteration>0 and history[-2]['true_newton_linear_residual']>.1)
        if rebuild:
            preconditioner,diagnostics=(preconditioner_builder or build_coarse)(residual,x,r,cutoff=cutoff)
        row.update(diagnostics);row['precondition_rebuilt']=rebuild
        force_rebuild=False
        def jv(v):
            epsilon=1e-7/max(np.linalg.norm(v),1e-100)
            return (residual(x+epsilon*v)-r)/epsilon
        h,z,y,linear=arnoldi(jv,r,preconditioner,limit=linear_limit,tolerance=linear_tolerance)
        newton=z@y;jn=jv(newton)
        true_linear=float(np.linalg.norm(r+jn)/norm)
        attempts=[dict(krylov_dimension=len(y),true_relative=true_linear)]
        if linear_refresh_threshold is not None and true_linear>linear_refresh_threshold and not rebuild:
            preconditioner,diagnostics=(preconditioner_builder or build_coarse)(residual,x,r,cutoff=cutoff)
            row.update(diagnostics);row['refreshed_for_accuracy']=True
            h,z,y,linear=arnoldi(jv,r,preconditioner,limit=linear_limit,tolerance=linear_tolerance)
            newton=z@y;jn=jv(newton);true_linear=float(np.linalg.norm(r+jn)/norm)
            attempts.append(dict(krylov_dimension=len(y),true_relative=true_linear))
        row['linear_attempts']=attempts
        row.update(krylov_dimension=len(y),arnoldi_linear_residual=linear,
                   true_newton_linear_residual=float(np.linalg.norm(r+jn)/norm),
                   newton_blocks=blocks(residual,newton),trials=[])
        if observer:observer(iteration,x.copy(),r.copy(),newton.copy())
        if linear_refresh_threshold is not None and true_linear>linear_refresh_threshold:
            return x,history,'linear_accuracy_limited'
        accepted=False
        for attempt in range(14):
            if mode=='hookstep':
                step,prediction,lam=hookstep(h,z,norm,radius,weights)
                alpha=None
            else:
                alpha=2.**(-attempt);step=alpha*newton;lam=0.
                prediction=.5*(norm*norm-np.linalg.norm(r+alpha*jn)**2)
            feasible=residual.feasible(x+step)
            trial=dict(radius=float(radius),alpha=alpha,lambda_value=lam,
                       feasible=bool(feasible),predicted_reduction=prediction,
                       step_blocks=blocks(residual,step))
            row['trials'].append(trial)
            if not feasible or prediction<=0:
                if mode=='hookstep':radius*=.25
                continue
            rr=residual(x+step);actual=.5*(norm*norm-np.linalg.norm(rr)**2)
            rho=actual/prediction
            trial.update(actual_reduction=float(actual),rho=float(rho),residual=float(np.linalg.norm(rr)))
            accept=(rho>.1 and actual>0) if mode=='hookstep' else (
                np.linalg.norm(rr)<(1-1e-4*alpha)*norm)
            if mode=='hookstep':
                if rho<.25:radius*=.25
                elif rho>.75 and np.linalg.norm(weights*step)>.8*radius:radius=min(2.,2*radius)
            if accept:
                row['accepted_trial']=attempt
                # Independent full-map directional check, not only Hessenberg prediction.
                true_prediction=.5*(norm*norm-np.linalg.norm(r+jv(step))**2)
                trial['true_predicted_reduction']=float(true_prediction)
                trial['true_rho']=float(actual/true_prediction) if true_prediction>0 else None
                x=x+step;r=rr;accepted=True
                if reanchor_threshold is not None and gauge_geometry(residual,x)['condition']>reanchor_threshold:
                    nongauge=r[:-2].copy()
                    reanchor(residual,x);r=residual(x)
                    row['reanchored']=True
                    row['reanchor_nongauge_change']=float(np.linalg.norm(r[:-2]-nongauge))
                    force_rebuild=True
                break
            if mode=='hookstep' and rho>=.25:radius*=.25
        if not accepted:return x,history,'globalization_stalled'
        if mode=='hookstep' and radius<1e-10:return x,history,'radius_floor_reached'
    return x,history,'iteration_budget_reached'
