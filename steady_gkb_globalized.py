"""Deep/recycled GKB with one basis per state and true-merit radius search."""
import time
import numpy as np
from steady_gkb import gkb,svd_trust
from steady_gkb_recycling import orthogonal_union


def radius_search(residual,x,r,q,jq,matrix,target,initial_radius,derivative,cauchy_direction,cauchy_response):
    """Select greatest actual decrease among probed, fully guarded candidates."""
    trials=[];candidates=[];probe_seconds=0.;gate_seconds=0.
    slope=float(r@cauchy_response);curvature=float(cauchy_response@cauchy_response)
    if slope>=0 or curvature<=0:return None,[],dict(probe_seconds=0.,gate_seconds=0.)
    def probe(radius):
        nonlocal probe_seconds
        start=time.perf_counter();y,pred,lam=svd_trust(matrix,target,radius)
        s=q@y;js=jq@y;feasible=bool(residual.feasible(x+s));rr=residual(x+s) if feasible else None
        actual=float((r@r-rr@rr)/2) if feasible else None;rho=actual/pred if feasible and pred>0 else -1.
        alpha=min(radius,-slope/curvature);pc=float(-alpha*slope-.5*alpha*alpha*curvature)
        norm=float(np.linalg.norm(s));p_response=float(-r@js-.5*(js@js))
        consistent=abs(pred-p_response)<.05*max(abs(pred),1e-100)
        eligible=bool(feasible and pred>0 and actual>0 and rho>.1 and pred>=pc*(1-1e-6) and norm<=radius*(1+1e-6) and consistent)
        row=dict(radius=radius,prediction=pred,actual=actual,rho=rho,step_norm=norm,lambda_value=lam,
          boundary=bool(lam>0 and norm>=radius*(1-1e-5)),feasible=feasible,eligible=eligible,
          cauchy_prediction=pc,response_prediction=p_response,checks=[],passed=False,
          nonlinear_defect=float(np.linalg.norm(rr-r-js)/max(np.linalg.norm(js),1e-100)) if feasible else None,
          residual=float(np.linalg.norm(rr)) if feasible else None)
        trials.append(row);candidates.append((s,rr));probe_seconds+=time.perf_counter()-start
        return len(trials)-1
    index=probe(initial_radius)
    for _ in range(12):
        t=trials[index]
        if t['eligible'] and t['rho']>=.25:break
        if t['radius']/2<1e-10:break
        index=probe(t['radius']/2)
    t=trials[index]
    if t['eligible'] and t['rho']>.8 and t['boundary']:
        for _ in range(12):
            enlarged=probe(trials[index]['radius']*1.5);new=trials[enlarged];old=trials[index]
            if not(new['eligible'] and new['actual']>old['actual'] and new['rho']>.5):break
            index=enlarged
            if not new['boundary']:break
    def select():
        nonlocal gate_seconds
        for i in sorted([i for i,t in enumerate(trials) if t['eligible'] and not t['checks']],key=lambda i:trials[i]['actual'],reverse=True):
            begin=time.perf_counter();t=trials[i];s,rr=candidates[i]
            for h in [1e-5,3e-6,1e-6]:
                js=derivative(s,h);pred=float(-r@js-.5*(js@js))
                t['checks'].append(dict(h=h,prediction=pred,relative=abs(pred-t['prediction'])/max(abs(t['prediction']),1e-100)))
            t['passed']=bool(all(c['prediction']>0 and c['relative']<.05 and t['actual']/c['prediction']>.1 for c in t['checks']))
            gate_seconds+=time.perf_counter()-begin
            if t['passed']:return i
        return None
    chosen=select()
    for _ in range(12):
        if chosen is not None:break
        radius=min(t['radius'] for t in trials)/2
        if radius<1e-10:break
        probe(radius);chosen=select()
    timing=dict(probe_seconds=probe_seconds,gate_seconds=gate_seconds,nonlinear_probes=len(trials))
    if chosen is None:return None,trials,timing
    return (chosen,*candidates[chosen]),trials,timing


def solve_outer(residual,jtv,initial,arm,old_basis=None,max_steps=20,radius=.0125,maximum=192,observer=None):
    """B carries only previous accepted state's fresh basis; never old J images."""
    if arm not in ['fresh','recycled']:raise ValueError(arm)
    if arm=='recycled' and old_basis is None:raise ValueError('Initial recycled basis required')
    x=initial.copy();r=residual(x);history=[];status='iteration_budget_reached'
    for outer in range(max_steps):
        if np.linalg.norm(r)<1e-7:status='residual_converged';break
        began=time.perf_counter()
        def derivative(v,h=1e-5):
            eps=h/max(np.linalg.norm(v),1e-100)
            return (residual(x+eps*v)-residual(x-eps*v))/(2*eps)
        size=maximum if arm=='fresh' else min(64,maximum)
        u,v,jv,B,ladder=gkb(derivative,lambda w:jtv(x,w),-r,size)
        fresh_seconds=time.perf_counter()-began
        recycled_seconds=0.;merge_seconds=0.;info=None
        if arm=='recycled':
            t=time.perf_counter();jold=np.column_stack([derivative(d) for d in old_basis.T]);recycled_seconds=time.perf_counter()-t
            t=time.perf_counter();q,jq,info,_=orthogonal_union(np.column_stack([v,old_basis]),np.column_stack([jv,jold]))
            u_small,small=np.linalg.qr(np.column_stack([-r,jq]),mode='reduced');matrix=small[:,1:];target=u_small.T@(-r);merge_seconds=time.perf_counter()-t
        else:
            q=v;jq=jv;matrix=B;target=np.zeros(len(B));target[0]=np.linalg.norm(r)
        row=dict(step=outer+1,initial_residual=float(np.linalg.norm(r)),initial_radius=radius,
          fresh_k=v.shape[1],rank=q.shape[1],fresh_seconds=fresh_seconds,recycled_response_seconds=recycled_seconds,
          merge_seconds=merge_seconds,basis_seconds=time.perf_counter()-began,
          u_orth=float(np.linalg.norm(u.T@u-np.eye(u.shape[1]))),v_orth=float(np.linalg.norm(q.T@q-np.eye(q.shape[1]))),
          bidiagonal_relative=float(np.linalg.norm(jv-u@B)/max(np.linalg.norm(jv),1e-100)),union=info,
          singular_values=np.linalg.svd(matrix,compute_uv=False).tolist())
        selected,trials,timing=radius_search(residual,x,r,q,jq,matrix,target,radius,derivative,v[:,0],jv[:,0])
        row.update(trials=trials,**timing)
        if selected is None:status='globalization_stalled'
        else:
            index,s,rr=selected;row['accepted_trial']=index;radius=trials[index]['radius'];row['next_radius']=radius
            if arm=='recycled':old_basis=v.copy()
            x=x+s;r=rr
        row['seconds']=time.perf_counter()-began;history.append(row)
        if observer:observer(x,r,history)
        if selected is None:break
    if np.linalg.norm(r)<1e-7:status='residual_converged'
    return x,history,status
