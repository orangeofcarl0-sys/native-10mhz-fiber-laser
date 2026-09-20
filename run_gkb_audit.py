"""Frozen latest endpoint: root/GKB/history ablation, no outer updates."""
import hashlib,json,time,zipfile
import numpy as np
from config import ROOT,PROJECT
from gpu_setup import cp
from discrete_adjoint import DiscreteAdjoint
from steady_descent_sources import physical_residual,BASE
from steady_support_lu import build_factored
from steady_hookstep import arnoldi
from steady_active_history import HistoryModel
from steady_seed_bank import prune_history
from steady_cauchy import cauchy_step,span_audit
from steady_gkb import gkb,svd_trust

begin=time.perf_counter()
folder=PROJECT/'results/steady_adjoint_continuation_20260920'
paths=[folder/'trajectory.npz',folder/'continuation.json',PROJECT/'results/steady_tail_20260919/tail_continuation.npz']
saved=np.load(paths[0]); prior=json.loads(paths[1].read_text(encoding='utf8'))
x=saved['states'][-1].copy(); bank=saved['bank'].T.copy()
f=physical_residual(np.load(paths[2]),gpu=True);adj=DiscreteAdjoint(f)
r,g=adj.value_and_vjp(x);d=-g/np.linalg.norm(g)
assert abs(np.linalg.norm(r)-.001096712976543593)<1e-13
radii=[.003125,.00625,.0125];dimensions=[1,2,4,8,12,16,24,32]
result=dict(initial_residual=float(np.linalg.norm(r)),physical_base=BASE,radii=radii,
 dimensions=dimensions,input_sha256={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
 baseline='Fresh C preconditioner at frozen endpoint; same original Arnoldi limit240/tol0.008; no current C gradient.',
 bank_rule='Prune pending13 once using A at radius0.00625, then freeze same12 across all arms/radii.',
 decision='B32 or C32 passes all guards and actual/A>1.5 at >=2 radii; no automatic outer run in this audit.',
 limitation='A single frozen state cannot exclude single-shooting or prove a formulation bottleneck.',
 source='https://web.stanford.edu/group/SOL/software/lsqr/',trials=[],ladder=[])
(ROOT/'protocol.json').write_text(json.dumps(result,indent=2),encoding='utf8')
with zipfile.ZipFile(ROOT/'executed_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PROJECT.glob('*.py')):z.write(p,p.name)
    z.write(PROJECT/'parameters/effective_parameters.json','parameters/effective_parameters.json')


def measured(call):
    cp.cuda.Stream.null.synchronize();t=time.perf_counter();answer=call()
    cp.cuda.Stream.null.synchronize();return answer,time.perf_counter()-t


def derivative(v,h=1e-5):
    eps=h/max(np.linalg.norm(v),1e-100)
    return (f(x+eps*v)-f(x-eps*v))/(2*eps)


def transpose(v):
    return adj.value_and_vjp(x,v)[1]


print('GKB AUDIT BENCHMARK',flush=True)
timing={}
for name,call in [('forward',lambda:f(x)),('Jv',lambda:derivative(d)),('JTv',lambda:transpose(r/np.linalg.norm(r)))]:
    values=[measured(call)[1] for _ in range(3)]
    timing[name]=dict(samples=values,median=float(np.median(values)))
print('GKB BUILD',flush=True)
(u,v,jv,B,ladder_time),timing['gkb_build']=measured(lambda:gkb(derivative,transpose,-r,32))
assert v.shape[1]==32
result['gkb_health']=dict(u_orth=float(np.linalg.norm(u.T@u-np.eye(u.shape[1]))),
 v_orth=float(np.linalg.norm(v.T@v-np.eye(v.shape[1]))),
 bidiagonal_relative=float(np.linalg.norm(jv-u@B)/np.linalg.norm(jv)),
 first_gradient_error=float(np.linalg.norm(v[:,0]-d)))
print('GKB HEALTH',result['gkb_health'],flush=True)
assert result['gkb_health']['bidiagonal_relative']<1e-5
print('BASELINE C BUILD',flush=True)
(pre,build),timing['c_build']=measured(lambda:build_factored(f,x,r,'C'))
(h,z,_,linear,q),timing['root_arnoldi']=measured(lambda:arnoldi(derivative,r,pre,limit=240,tolerance=.008,return_basis=True))
jd=derivative(d)
jbank,timing['history_responses']=measured(lambda:np.column_stack([derivative(a) for a in bank.T]))
model=HistoryModel(np.column_stack([z,bank,d]),np.column_stack([q@h,jbank,jd]),r,z.shape[1])
selected,events=prune_history(model,bank.shape[1],12,bank.shape[1],.00625)
bank=bank[:,selected];jbank=jbank[:,selected]
result.update(history_ids=[prior['final_pending_bank_ids'][i] for i in selected],pruning=events,
 root_dimension=z.shape[1],root_linear_residual=linear,core_build=build,
 gradient_coverage=span_audit(z,d,np.ones_like(d))[0],timing=timing)
models={'A':HistoryModel(np.column_stack([z,d,bank]),np.column_stack([q@h,jd,jbank]),r,z.shape[1]+13)}
steps={}


def assess(name,k,radius,s,p,lam):
    feasible=bool(f.feasible(x+s));rr=f(x+s)
    actual=float((r@r-rr@rr)/2)
    checks=[]
    for eps in [1e-5,3e-6,1e-6]:
        js=derivative(s,eps);pred=float(-r@js-.5*(js@js))
        checks.append(dict(h=eps,prediction=pred,relative=abs(pred-p)/max(abs(p),1e-100)))
    pc=cauchy_step(r,d,jd,radius,np.ones_like(d))[1]
    passed=bool(feasible and p>=pc*(1-1e-6) and np.linalg.norm(s)<=radius*(1+1e-6)
       and actual>0 and actual/p>.1 and all(a['prediction']>0 and a['relative']<.05 for a in checks))
    row=dict(arm=name,k=k,radius=radius,prediction=p,actual=actual,rho=actual/p,
             residual=float(np.linalg.norm(rr)),step_norm=float(np.linalg.norm(s)),lambda_value=lam,
             cauchy_prediction=pc,feasible=feasible,passed=passed,checks=checks)
    result['trials'].append(row);steps[f'{name}_{k}_{radius}']=s
    print('TRIAL',name,k,radius,'pred',p,'actual',actual,'rho',actual/p,'pass',passed,flush=True)


for radius in radii:
    assess('A',z.shape[1],radius,*models['A'].full_step([],radius))
for k in range(1,33):
    # The nominal bidiagonal model is used for B, with independent Js gates.
    target=np.zeros(k+1);target[0]=np.linalg.norm(r)
    for radius in radii:
        y,p,lam=svd_trust(B[:k+1,:k],target,radius)
        result['ladder'].append(dict(k=k,radius=radius,prediction=p,
                                    build_seconds=ladder_time[k-1]['seconds']))
        if k in dimensions:
            assess('B',k,radius,v[:,:k]@y,p,lam)
    if k not in dimensions:continue
    for name,basis,response in [('C',np.column_stack([v[:,:k],bank]),np.column_stack([jv[:,:k],jbank])),
                                ('D',np.column_stack([v[:,:k],bank,z]),np.column_stack([jv[:,:k],jbank,q@h]))]:
        model=HistoryModel(basis,response,r,basis.shape[1])
        for radius in radii:assess(name,k,radius,*model.full_step([],radius))
    (ROOT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
result['decision_counts']={name:sum(t['passed'] and t['actual']/next(a['actual'] for a in result['trials'] if a['arm']=='A' and a['radius']==t['radius'])>1.5
    for t in result['trials'] if t['arm']==name and t['k']==32) for name in ['B','C']}
result['upgrade_gate']=any(n>=2 for n in result['decision_counts'].values())
result['seconds']=time.perf_counter()-begin
np.savez_compressed(ROOT/'audit_vectors.npz',x=x,r=r,g=g,U=u,V=v,JV=jv,B=B,bank=bank,Jbank=jbank,Z=z,JZ=q@h,**steps)
(ROOT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('AUDIT COMPLETE',result['decision_counts'],result['upgrade_gate'],flush=True)
