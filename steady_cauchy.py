"""Cauchy enrichment in the existing state metric; no physical-map changes."""
import numpy as np
from steady_hookstep import hookstep

def span_audit(z,d,weights):
    u,s,vh=np.linalg.svd(weights[:,None]*z,full_matrices=False)
    target=weights*d;norm=np.linalg.norm(target);rows=[];coefficients={}
    for label,cutoff in [('raw',1e-14),('kept',1e-12)]:
        keep=s>s[0]*cutoff;q=u[:,keep];projected=q@(q.T@target)
        coefficients[label]=vh[keep].T@((q.T@target)/s[keep])
        error=np.linalg.norm(target-projected)/norm;coverage=np.linalg.norm(q.T@target)/norm
        rows.append(dict(span=label,relative_cutoff=cutoff,rank=int(keep.sum()),epsilon=float(error),
                         coverage=float(coverage),partition=float(error**2+coverage**2)))
    return rows,coefficients,s

def cauchy_step(r,d,jd,radius,weights):
    slope=float(np.dot(r,jd));curvature=float(np.dot(jd,jd));metric=np.linalg.norm(weights*d)
    if slope>=0 or curvature<=0 or metric<=0:raise ValueError('A verified descent direction is required')
    alpha=min(-slope/curvature,radius/metric)
    step=alpha*d;prediction=float(-alpha*slope-.5*alpha*alpha*curvature)
    return step,prediction,float(alpha)

def augmented_model(h,z,v,d,jd):
    # Reorthogonalize the extra OUTPUT response; v[:,0]=-R/beta fixes target sign.
    remainder=jd.copy();q=np.zeros(v.shape[1])
    for _ in range(2):
        correction=v.T@remainder;q+=correction;remainder-=v@correction
    gamma=np.linalg.norm(remainder)
    ha=np.zeros((h.shape[0]+1,h.shape[1]+1));ha[:-1,:-1]=h;ha[:-1,-1]=q;ha[-1,-1]=gamma
    return ha,np.column_stack([z,d]),float(gamma)

def augmented_step(h,z,v,r,d,jd,radius,weights):
    c,pc,alpha=cauchy_step(r,d,jd,radius,weights)
    ha,za,gamma=augmented_model(h,z,v,d,jd)
    step,prediction,lam=hookstep(ha,za,np.linalg.norm(r),radius,weights)
    passed=bool(np.isfinite(prediction) and prediction>=pc*(1-1e-6) and np.linalg.norm(weights*step)<=radius*(1+1e-6))
    # Keep the failed raw result visible; never call a fallback an augmentation pass.
    chosen=step if passed else c
    return chosen,dict(raw_model_pass=passed,raw_prediction=float(prediction),cauchy_prediction=pc,
        selected_prediction=float(prediction if passed else pc),fallback=not passed,
        alpha_cauchy=alpha,output_orthogonal_norm=gamma,lambda_value=lam),step
