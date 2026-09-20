"""Frozen latest state: depth192 and current-response recycled old64."""
import hashlib,json,time,zipfile
import numpy as np
from config import ROOT,PROJECT
from gpu_setup import cp
from discrete_adjoint import DiscreteAdjoint
from steady_descent_sources import physical_residual,BASE
from steady_gkb import gkb,svd_trust
from steady_gkb_recycling import orthogonal_union

begin=time.perf_counter();folder=PROJECT/'results/steady_pure_gkb_20260920'
paths=[folder/'trajectory.npz',folder/'continuation.json',PROJECT/'results/steady_tail_20260919/tail_continuation.npz']
states=np.load(paths[0])['states'];x=states[-1].copy();old=states[-2].copy()
prior=json.loads(paths[1].read_text(encoding='utf8'));radius=prior['history'][-1]['next_radius'];assert radius==.025
f=physical_residual(np.load(paths[2]),gpu=True);adj=DiscreteAdjoint(f)
r=f(x);rold=f(old);assert abs(np.linalg.norm(r)-.0009746756121000674)<1e-13
checkpoints=list(range(64,193,16))
a=dict(initial_residual=float(np.linalg.norm(r)),old_residual=float(np.linalg.norm(rold)),radius=radius,
 physical_base=BASE,checkpoints=checkpoints,trials=[],timing={},
 input_sha256={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
 old_state_definition='states[-2], the state from which accepted step40 was constructed; current=states[-1]',
 decision='Two consecutive16-column gains<1%; recycling >90% prediction and <=80% warm cost vs each of K128/K192, with guards. No outer continuation.',
 published_vectors='Small B, states, old/new candidate steps. Full basis archive retained locally; deterministic reconstruction script provided.')
(ROOT/'protocol.json').write_text(json.dumps(a,indent=2),encoding='utf8')
with zipfile.ZipFile(ROOT/'executed_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PROJECT.glob('*.py')):z.write(p,p.name)
    z.write(PROJECT/'parameters/effective_parameters.json','parameters/effective_parameters.json')


def measured(call):
    cp.cuda.Stream.null.synchronize();t=time.perf_counter();answer=call()
    cp.cuda.Stream.null.synchronize();return answer,time.perf_counter()-t


def derivative(at,v,h=1e-5):
    eps=h/max(np.linalg.norm(v),1e-100)
    return (f(at+eps*v)-f(at-eps*v))/(2*eps)


def build(at,residual,maximum,label):
    calls=0
    def transpose(v):
        nonlocal calls
        calls+=1
        if calls%16==0:print(label,'COLUMN',calls,flush=True)
        return adj.value_and_vjp(at,v)[1]
    return gkb(lambda v:derivative(at,v),transpose,-residual,maximum)


adj.value_and_vjp(x)
(uo,vo,jvo,Bo,old_ladder),a['timing']['old_rebuild']=measured(lambda:build(old,rold,64,'OLD'))
target=np.zeros(65);target[0]=np.linalg.norm(rold)
yo,po,_=svd_trust(Bo,target,radius)
old_prediction=prior['history'][-1]['ladder'][-1]['prediction']
a['old_reproduction_relative']=abs(po/old_prediction-1);assert a['old_reproduction_relative']<1e-6
(u,v,jv,B,ladder),a['timing']['fresh192']=measured(lambda:build(x,r,192,'FRESH'))
assert v.shape[1]==192
a['health']=dict(u_orth=float(np.linalg.norm(u.T@u-np.eye(u.shape[1]))),
 v_orth=float(np.linalg.norm(v.T@v-np.eye(v.shape[1]))),bidiagonal_relative=float(np.linalg.norm(jv-u@B)/np.linalg.norm(jv)))
assert a['health']['bidiagonal_relative']<1e-5
jold,a['timing']['recompute_old_responses']=measured(lambda:np.column_stack([derivative(x,d) for d in vo.T]))
saved=dict(x=x,old=old,r=r,rold=rold,B=B,Bo=Bo,old_step=vo@yo)
d=v[:,0];jd=jv[:,0];alpha=min(radius,-r@jd/(jd@jd));pc=float(-alpha*(r@jd)-.5*alpha**2*(jd@jd))


def assess(name,step,p,lam,cost,spectrum):
    feasible=bool(f.feasible(x+step));rr=f(x+step)
    actual=float((r@r-rr@rr)/2);checks=[]
    for eps in [1e-5,3e-6,1e-6]:
        js=derivative(x,step,eps);pred=float(-r@js-.5*(js@js))
        checks.append(dict(h=eps,prediction=pred,relative=abs(pred-p)/max(abs(p),1e-100)))
    physical=bool(feasible and actual>0 and p>0 and actual/p>.1 and np.linalg.norm(step)<=radius*(1+1e-6)
        and all(q['prediction']>0 and q['relative']<.05 for q in checks))
    passed=bool(physical and p>=pc*(1-1e-6))
    row=dict(name=name,prediction=p,actual=actual,rho=actual/p,step_norm=float(np.linalg.norm(step)),
       lambda_value=lam,residual=float(np.linalg.norm(rr)),feasible=feasible,physical_pass=physical,
       cauchy_prediction=pc,passed=passed,checks=checks,cost_seconds=cost,singular_values=spectrum.tolist())
    a['trials'].append(row);saved['step_'+name]=step
    print('CANDIDATE',name,'prediction',p,'actual',actual,'rho',actual/p,'passed',passed,flush=True)
    return row


previous=None;streak=0;first=None
for k in checkpoints:
    target=np.zeros(k+1);target[0]=np.linalg.norm(r)
    (y,p,lam),solve_time=measured(lambda:svd_trust(B[:k+1,:k],target,radius))
    row=assess('K'+str(k),v[:,:k]@y,p,lam,ladder[k-1]['seconds']+solve_time,np.linalg.svd(B[:k+1,:k],compute_uv=False))
    row.update(k=k,build_seconds=ladder[k-1]['seconds'],solve_seconds=solve_time)
    if previous is not None:
        row.update(absolute_increment=p-previous,relative_increment=(p-previous)/p)
        streak=streak+1 if 0<=row['relative_increment']<.01 else 0
        if streak>=2 and first is None:first=k
    previous=p
    (ROOT/'audit.json').write_text(json.dumps(a,indent=2),encoding='utf8')


def merged_model(basis,response):
    q,jq,info,transform=orthogonal_union(basis,response)
    out,small=np.linalg.qr(np.column_stack([-r,jq]),mode='reduced')
    target=out.T@(-r)
    y,p,lam=svd_trust(small[:,1:],target,radius)
    info.update(orth_error=float(np.linalg.norm(q.T@q-np.eye(q.shape[1]))))
    return q@y,p,lam,np.linalg.svd(small[:,1:],compute_uv=False),info,small,target


(sr,pr,lr,sigr,infoR,smallR,targetR),costR=measured(lambda:merged_model(vo,jold))
assess('R64',sr,pr,lr,a['timing']['recompute_old_responses']+costR,sigr)
(sc,pcc,lc,sigc,infoC,smallC,targetC),costC=measured(lambda:merged_model(np.column_stack([v[:,:64],vo]),np.column_stack([jv[:,:64],jold])))
warm=ladder[63]['seconds']+a['timing']['recompute_old_responses']+costC
assess('K64+R64',sc,pcc,lc,warm,sigc)
a['merge']=infoC;a['recycled']=infoR;a['timing'].update(recycled_solve=costR,union_merge_solve=costC,
     union_warm=warm,union_cold=warm+a['timing']['old_rebuild'])
a['first_saturation_checkpoint']=first
a['recycling_comparisons']={}
for k in [128,192]:
    fresh=next(t for t in a['trials'] if t['name']=='K'+str(k))
    coverage=pcc/fresh['prediction'];ratio=warm/fresh['cost_seconds']
    a['recycling_comparisons'][str(k)]=dict(model_coverage=coverage,cost_ratio=ratio,
       actual_ratio=a['trials'][-1]['actual']/fresh['actual'] if fresh['actual']>0 else None,supported=bool(coverage>.9 and ratio<=.8 and a['trials'][-1]['passed']))
a['union_gain_over_K64']=pcc/a['trials'][0]['prediction'];a['seconds']=time.perf_counter()-begin
saved.update(smallR=smallR,targetR=targetR,smallC=smallC,targetC=targetC)
np.savez_compressed(ROOT/'candidate_vectors.npz',**saved)
# Local numerical archive: public reproduction rebuilds these large bases.
np.savez_compressed(ROOT/'local_full_bases.npz',U=u,V=v,JV=jv,Vold=vo,JcurrentVold=jold)
a['local_basis_sha256']=hashlib.sha256((ROOT/'local_full_bases.npz').read_bytes()).hexdigest()
(ROOT/'audit.json').write_text(json.dumps(a,indent=2),encoding='utf8')
print('DEPTH AUDIT COMPLETE',a['recycling_comparisons'],'saturation',first,flush=True)
