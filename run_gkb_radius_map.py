"""Frozen saved GKB spaces: radius re-solves and finite-step ray defects."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual
from steady_gkb import svd_trust
from steady_gkb_recycling import orthogonal_union

source=Path(os.environ['LASER_BASIS_DIR'])
paths=[source/'local_full_bases.npz',source/'candidate_vectors.npz',source/'audit.json']
archive=np.load(paths[0]);prior=np.load(paths[1]);audit=json.loads(paths[2].read_text(encoding='utf8'))
x=prior['x'];r=prior['r'];V=archive['V'];JV=archive['JV'];B=prior['B']
f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True)
assert np.linalg.norm(f(x)-r)<1e-14
radii=[.003125,.00625,.0125,.01875,.025];depths=list(range(64,193,16))
protocol=dict(initial_residual=float(np.linalg.norm(r)),radii=radii,depths=depths,
 input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
 derivative_calls=0,physical_base=audit['physical_base'],
 scope='Frozen basis and state. 45 fresh radius +5 recycled radius +15 ray. No outer, no new Jv/JTv. Ray K96 beyond original norm is extrapolation.')
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
rows=[];vectors=dict(x=x,r=r);start=time.perf_counter()
q,jq,info,_=orthogonal_union(np.column_stack([V[:,:64],archive['Vold']]),np.column_stack([JV[:,:64],archive['JcurrentVold']]))
out,small=np.linalg.qr(np.column_stack([-r,jq]),mode='reduced');targetC=out.T@(-r)
health=dict(v_orth=float(np.linalg.norm(V.T@V-np.eye(192))),
 jv_ub_relative=float(np.linalg.norm(JV-archive['U']@B)/np.linalg.norm(JV)),union_rank=info['rank'])

def assess(name,kind,k,radius,step,js,pred,lam,extra):
    rr=f(x+step);actual=float((r@r-rr@rr)/2);defect=rr-r-js
    norm=float(np.linalg.norm(step));jd=JV[:,0];alpha=min(radius,max(0.,-r@jd/(jd@jd)))
    cauchy=float(-alpha*r@jd-.5*alpha**2*(jd@jd))
    linear=float(-r@js-.5*(js@js));agreement=abs(linear-pred)/max(abs(pred),1e-100)
    feasible=bool(f.feasible(x+step));rho=actual/pred if pred>0 else None
    row=dict(name=name,kind=kind,k=k,radius=radius,prediction=pred,actual=actual,rho=rho,
      step_norm=norm,lambda_value=lam,feasible=feasible,
      nonlinear_defect=float(np.linalg.norm(defect)/max(np.linalg.norm(js),1e-100)),
      defect_field=float(np.linalg.norm(defect[:65536])),defect_population=float(np.linalg.norm(defect[65536:-2])),
      defect_gauge=float(np.linalg.norm(defect[-2:])),response_norm=float(np.linalg.norm(js)),
      prediction_response_relative=agreement,cauchy_prediction=cauchy,
      passed=bool(feasible and pred>0 and actual>0 and rho>.1 and agreement<.05 and pred>=cauchy*(1-1e-6)),**extra)
    assert agreement<.05
    rows.append(row);vectors[name+'_step']=step;vectors[name+'_js']=js;vectors[name+'_residual']=rr
    print(name,'actual',actual,'rho',rho,'defect',row['nonlinear_defect'],flush=True)

for k in depths:
    b=B[:k+1,:k];target=np.zeros(k+1);target[0]=np.linalg.norm(r)
    for i,radius in enumerate(radii):
        y,p,lam=svd_trust(b,target,radius);step=V[:,:k]@y;js=JV[:,:k]@y
        kkt=np.linalg.norm(b.T@(b@y-target)+lam*y)/max(np.linalg.norm(b.T@target),1e-100)
        assert kkt<1e-6 and np.linalg.norm(step)<=radius*(1+1e-6)
        assess(f'K{k}_D{i}','trust',k,radius,step,js,p,lam,dict(kkt_relative=float(kkt),boundary=bool(lam>0)))
        if i==4:
            reference=next(t for t in audit['trials'] if t['name']==f'K{k}')
            assert abs(rows[-1]['actual']-reference['actual'])<1e-15
            assert np.linalg.norm(step-prior[f'step_K{k}'])<1e-10
for i,radius in enumerate(radii):
    y,p,lam=svd_trust(small[:,1:],targetC,radius)
    assess(f'UNION_D{i}','recycled',128,radius,q@y,jq@y,p,lam,dict(boundary=bool(lam>0)))
    if i==4:assert np.linalg.norm(q@y-prior['step_K64+R64'])<1e-9
for k in [96,128,192]:
    original=next(t for t in rows if t['name']==f'K{k}_D4');s=vectors[f'K{k}_D4_step'];js=vectors[f'K{k}_D4_js'];norm=np.linalg.norm(s)
    for i,length in enumerate(radii[:4]+[float(norm)]):
        alpha=length/norm;response=alpha*js;pred=float(-r@response-.5*(response@response))
        assess(f'RAY{k}_{i}','ray',k,length,alpha*s,response,pred,None,dict(alpha=float(alpha),extrapolated=bool(alpha>1+1e-10)))
result=dict(protocol=protocol,health=health,rows=rows,seconds=time.perf_counter()-start)
np.savez_compressed(ROOT/'candidates.npz',**vectors)
(ROOT/'radius_map.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('COMPLETE',len(rows),flush=True)
