"""Frozen period-one local geometry; no nonlinear outer update."""
import hashlib,json,time,zipfile
import numpy as np
from config import ROOT,PROJECT
from discrete_adjoint import DiscreteAdjoint
from steady_descent_sources import physical_residual,BASE
from steady_gkb import gkb
from local_geometry import curvature_probe

source=PROJECT/'results/steady_gkb_outer_ab_20260923/fresh/trajectory.npz'
states=np.load(source)['states'];x=states[-1].copy();last_step=x-states[-2]
f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True);adj=DiscreteAdjoint(f)
r=f(x);ra,g=adj.value_and_vjp(x);assert np.linalg.norm(r-ra)<1e-13
assert abs(np.linalg.norm(r)-.0009315468344505816)<1e-13
checkpoints=[64,96,128,192,256,320,384];start=time.perf_counter()
protocol=dict(initial_residual=float(np.linalg.norm(r)),physical_base=BASE,checkpoints=checkpoints,
 input_sha256={str(source.relative_to(PROJECT)):hashlib.sha256(source.read_bytes()).hexdigest()},
 scales=[1e-4,3e-5,1e-5,3e-6,1e-6],reference_h=1e-5,pseudoinverse_cutoff=1e-12,
 scope='Stationarity, unconstrained linear diagnostic only, directional full Hessian. No outer step.',
 pilot_gate='chi>=.01 and eta384<=.1 and validated kappa>=.5; OR eta>=.5 with full normal ratio<.001 and last2 eta gains<.001.')
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
with zipfile.ZipFile(ROOT/'executed_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in PROJECT.glob('*.py'):z.write(p,p.name)
    z.write(PROJECT/'parameters/effective_parameters.json','parameters/effective_parameters.json')

def jv(v,h=1e-5):
    eps=h/max(np.linalg.norm(v),1e-100)
    return (f(x+eps*v)-f(x-eps*v))/(2*eps)

def jt(v):return adj.value_and_vjp(x,v)[1]

# Independent starts distinguish a stable dominant Ritz estimate from one start.
power=[];rng=np.random.default_rng(20260923)
for seed in range(3):
    q=rng.normal(size=len(x));q/=np.linalg.norm(q);history=[]
    for i in range(60):
        jq=jv(q);z=jt(jq);mu=float(q@z);error=float(np.linalg.norm(z-mu*q)/max(abs(mu),1e-100))
        history.append(dict(iteration=i+1,sigma=float(np.sqrt(mu)),eigen_residual=error))
        if i>=19 and error<1e-5:break
        q=z/np.linalg.norm(z)
    power.append(dict(start=seed,history=history));print('SIGMA',seed,history[-1],flush=True)
calls=0

def counted_jt(v):
    global calls
    calls+=1
    if calls%32==0:print('LINEAR GKB',calls,flush=True)
    return jt(v)

began=time.perf_counter();U,V,JV,B,ladder=gkb(jv,counted_jt,-r,384);basis_seconds=time.perf_counter()-began
assert V.shape[1]==384
result=dict(protocol=protocol,power=power,basis_seconds=basis_seconds,
 health=dict(v_orth=float(np.linalg.norm(V.T@V-np.eye(384))),u_orth=float(np.linalg.norm(U.T@U-np.eye(385))),jv_ub_relative=float(np.linalg.norm(JV-U@B)/np.linalg.norm(JV))),linear=[],curvature=[])
saved=dict(x=x,r=r,g=g,B=B,last_step=last_step);steps={}
for k in checkpoints:
    small=B[:k+1,:k];p,s,qt=np.linalg.svd(small,full_matrices=False);target=np.zeros(k+1);target[0]=np.linalg.norm(r)
    keep=s>s[0]*1e-12;coeff=np.divide(p.T@target,s,out=np.zeros(k),where=keep);y=qt.T@coeff
    step=V[:,:k]@y;response=JV[:,:k]@y;linear_residual=r+response;normal=jt(linear_residual);direct=jv(step)
    row=dict(k=k,eta=float(np.linalg.norm(linear_residual)/np.linalg.norm(r)),
       eta_small=float(np.linalg.norm(target-small@y)/np.linalg.norm(r)),
       full_normal_ratio=float(np.linalg.norm(normal)/np.linalg.norm(g)),
       projected_normal_ratio=float(np.linalg.norm(V[:,:k].T@normal)/np.linalg.norm(g)),
       direct_response_relative=float(np.linalg.norm(direct-response)/np.linalg.norm(response)),
       step_norm=float(np.linalg.norm(step)),retained=int(keep.sum()),singular_values=s.tolist(),seconds=ladder[k-1]['seconds'])
    steps[k]=step;saved[f'linear_step_{k}']=step;saved[f'linear_residual_{k}']=linear_residual
    result['linear'].append(row);print('LINEAR',k,row['eta'],row['full_normal_ratio'],row['step_norm'],flush=True)
    (ROOT/'geometry.json').write_text(json.dumps(result,indent=2),encoding='utf8')
sigmas=[v['history'][-1]['sigma'] for v in power]+[result['linear'][-1]['singular_values'][0]]
sigma=max(sigmas);chi=float(np.linalg.norm(g)/(sigma*np.linalg.norm(r)))
result['stationarity']=dict(gradient_norm=float(np.linalg.norm(g)),sigma_estimate=sigma,chi=chi,
 sigma_estimate_spread=float((max(sigmas)-min(sigmas))/sigma),power_converged=bool(all(p['history'][-1]['eigen_residual']<1e-5 for p in power)),
 warning='Estimated norm, not rigorous upper bound. Small chi does not prove a nonzero minimum or absence of roots.')
directions={'full_gradient':-g,'K64':steps[64],'K96':steps[96],'K192':steps[192],'last_accepted':last_step}
for i in range(3):directions[f'tail_random_{i+1}']=V[:,320:384]@rng.normal(size=64)
all_hv={}
for name,d in directions.items():
    d=d/np.linalg.norm(d)
    assert all(f.feasible(x+h*d) and f.feasible(x-h*d) for h in protocol['scales'])
    rows,vec,transpose_error=curvature_probe(adj.value_and_vjp,x,r,d,jv,jt)
    reference=rows[2];fine_errors=[abs(rows[i]['full']-reference['full'])/reference['gn'] for i in [3,4]]
    row=dict(name=name,scales=rows,transpose_quadratic_relative=transpose_error,fine_scale_errors=fine_errors,validated=bool(max(fine_errors)<.01))
    result['curvature'].append(row)
    for key in ['direction','jv','gn']:saved[name+'_'+key]=vec[key]
    saved[name+'_hv']=vec['hv'][2];all_hv[name]=vec['hv']
    print('CURVATURE',name,'kappa',reference['kappa'],'full',reference['full'],'validated',row['validated'],flush=True)
    (ROOT/'geometry.json').write_text(json.dumps(result,indent=2),encoding='utf8')
last=result['linear'][-1];eta=result['linear'];marginals=[(eta[i-1]['eta']-eta[i]['eta'])/eta[i-1]['eta'] for i in [-2,-1]]
large_curvature=any(t['validated'] and t['scales'][2]['kappa']>=.5 for t in result['curvature'])
resolved_floor=bool(last['eta']>=.5 and last['full_normal_ratio']<1e-3 and all(0<=v<1e-3 for v in marginals))
curved_solvable=bool(result['stationarity']['power_converged'] and chi>=.01 and last['eta']<=.1 and large_curvature)
result['pilot_decision']=dict(curved_solvable=curved_solvable,resolved_high_floor=resolved_floor,trigger=curved_solvable or resolved_floor,
 last_eta_marginals=marginals,large_validated_curvature=large_curvature)
result['seconds']=time.perf_counter()-start
np.savez_compressed(ROOT/'vectors.npz',**saved)
np.savez_compressed(ROOT/'local_hessians.npz',**all_hv)
np.savez_compressed(ROOT/'local_basis.npz',V=V,JV=JV,B=B)
result['local_basis_sha256']=hashlib.sha256((ROOT/'local_basis.npz').read_bytes()).hexdigest()
(ROOT/'geometry.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('GEOMETRY COMPLETE',result['stationarity'],result['pilot_decision'],flush=True)
