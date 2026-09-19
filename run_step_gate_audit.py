"""Fixed-state directional plateau and trusted candidate audit."""
import json, hashlib
import numpy as np
from config import ROOT, PROJECT
from steady_parameters import parameter_residual
from steady_support_lu import build_factored
from steady_hookstep import arnoldi
from steady_cauchy import cauchy_step, augmented_step

hs=[1e-4,3e-5,1e-5,3e-6,1e-6,3e-7]
base=PROJECT/'results/steady_augmented_pair_20260920'
template=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
result={'scales':hs,'arms':{},'candidates':[]}
for arm in ['A','B']:
    x=np.load(base/f'{arm}_final.npz')['x']; d=np.load(base/f'{arm}_full_budget_direction.npz')['newton']
    f=parameter_residual(template,'pump',(.0275-.05)/.005,gpu=True);r=f(x)
    def derivative(v,h=1e-5):
        eps=h/max(np.linalg.norm(v),1e-100)
        return (f(x+eps*v)-f(x-eps*v))/(2*eps)
    ds=[derivative(d,h) for h in hs]
    rows=[]
    for h,j in zip(hs,ds):
        half=derivative(d,h/2);rich=(4*half-j)/3
        rows.append(dict(h=h,eta=float(np.linalg.norm(r+j)/np.linalg.norm(r)),derivative_norm=float(np.linalg.norm(j)),numerator_norm=float(2*h*np.linalg.norm(j)/np.linalg.norm(d)),richardson_eta=float(np.linalg.norm(r+rich)/np.linalg.norm(r)),half_change_over_R=float(np.linalg.norm(half-j)/np.linalg.norm(r))))
    result['arms'][arm]=dict(residual=float(np.linalg.norm(r)),newton_norm=float(np.linalg.norm(d)),rows=rows,pairwise_change_over_R=[[float(np.linalg.norm(a-b)/np.linalg.norm(r)) for b in ds] for a in ds])
    print(arm,rows,flush=True)
    if arm=='B':
        cache={};print('BUILD B',flush=True);pre,info=build_factored(f,x,r,'C',descent=cache)
        H,Z,y,linear,V=arnoldi(derivative,r,pre,limit=240,tolerance=.008,return_basis=True)
        result['basis']=dict(dimension=len(y),internal=linear,build=info)
        direction=-cache['lift'](cache['gradient']);direction/=np.linalg.norm(direction);jd=derivative(direction)
        for radius in [.00156,.003125,.00625]:
            sc,pc,alpha=cauchy_step(r,direction,jd,radius,np.ones_like(x))
            selected,guard,raw=augmented_step(H,Z,V,r,direction,jd,radius,np.ones_like(x))
            for kind,s,pred in [('cauchy',sc,pc),('raw_augmented',raw,guard['raw_prediction'])]:
                rr=f(x+s);actual=float((r@r-rr@rr)/2);checks=[]
                for h in hs:
                    js=derivative(s,h);prediction=float(-r@js-.5*(js@js))
                    checks.append(dict(h=h,prediction=prediction,discrepancy=abs(pred-prediction)/max(abs(pred),1e-100)))
                row=dict(radius=radius,kind=kind,prediction=pred,actual=actual,rho=actual/pred,step_norm=float(np.linalg.norm(s)),checks=checks,guard=guard)
                result['candidates'].append(row);print('CANDIDATE',radius,kind,row['rho'],[v['discrepancy'] for v in checks],flush=True)
    (ROOT/'audit.json').write_text(json.dumps(result,indent=2))
result['continuation_allowed']=any(v['kind']=='raw_augmented' and v['guard']['raw_model_pass'] and v['rho']>.1 and all(c['discrepancy']<.05 for c in v['checks'] if c['h'] in [3e-6,1e-6]) for v in result['candidates'])
(ROOT/'audit.json').write_text(json.dumps(result,indent=2));print('ALLOWED',result['continuation_allowed'],flush=True)
