"""Frozen S3 endpoint: seven fresh histories, five groups, three radii; no continuation."""
import hashlib,json,time,zipfile
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual,annulus_coordinates,streamed_gradient,BASE
from steady_support_lu import build_factored
from steady_hookstep import arnoldi
from steady_cauchy import cauchy_step
from steady_active_history import HistoryModel,greedy_history

start=time.perf_counter();radii=[.00156,.003125,.00625];scales=[1e-5,3e-6,1e-6]
source=PROJECT/'results/steady_descent_sources_20260920'
template_path=PROJECT/'results/steady_tail_20260919/tail_continuation.npz'
inputs=[source/'continuation_final.npz',source/'S3_fresh_history.npz',template_path]
with zipfile.ZipFile(ROOT/'executed_sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(PROJECT.glob('*.py')):archive.write(path,path.name)
    archive.write(PROJECT/'parameters/effective_parameters.json','parameters/effective_parameters.json')
x=np.load(inputs[0])['x'];bank=np.load(inputs[1]);dh=bank['directions'];assert dh.shape==(7,len(x))
f=physical_residual(np.load(template_path),gpu=True);r=f(x)
assert abs(np.linalg.norm(r)-.0011883945817279438)<1e-14
result=dict(input_sha256={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    initial_residual=float(np.linalg.norm(r)),physical_base=BASE,radii=radii,
    history_outer_steps=[1,7,11,14,17,18,28],directions={},greedy=[],trials=[],
    contract='Single frozen state, forced G2/G3/G4 ladder, separate 5% stop diagnostic; no continuation or parameter release.')
def save(): (ROOT/'active_history.json').write_text(json.dumps(result,indent=2),encoding='utf8')
save()
def derivative(d,h=1e-5):
    eps=h/max(np.linalg.norm(d),1e-100)
    return (f(x+eps*d)-f(x-eps*d))/(2*eps)
def checked_direction(d):
    js=[derivative(d,h) for h in scales];j=js[0]
    return j,dict(slope=float(r@j),response_norm=float(np.linalg.norm(j)),
        relative_checks=[float(np.linalg.norm(v-j)/max(np.linalg.norm(j),1e-100)) for v in js])
print('CURRENT C BUILD',flush=True);cache={}
pre,build=build_factored(f,x,r,'C',descent=cache)
dc=-cache['lift'](cache['gradient']);dc/=np.linalg.norm(dc)
jc,result['directions']['core']=checked_direction(dc)
print('CURRENT ARNOLDI',flush=True)
H,Z,y,linear,V=arnoldi(derivative,r,pre,limit=240,tolerance=.008,return_basis=True)
result['common_basis']=dict(dimension=len(y),internal_error=linear,build=build)
jh=[]
for i,d in enumerate(dh):
    j,info=checked_direction(d);jh.append(j);result['directions'][str(i)]=info
jh=np.array(jh)
result['rank']={}
for name,matrix in [('D',dh.T),('JD',jh.T)]:
    singular=np.linalg.svd(matrix,compute_uv=False);relative=singular/singular[0]
    result['rank'][name]=dict(singular=singular.tolist(),relative=relative.tolist(),
        numerical_ranks={str(t):int(np.sum(relative>t)) for t in [1e-2,1e-3,1e-4,1e-6,1e-8,1e-12]})
result['history_gram']=(dh@dh.T).tolist();save()
print('FRESH ANNULUS ORACLE',flush=True)
_,lift,size=annulus_coordinates(f.n,f.cells)
ga,gforward=streamed_gradient(f,x,r,lift,size);da=-lift(ga);da/=np.linalg.norm(da)
ja,result['directions']['annulus']=checked_direction(da)
result['annulus']=dict(gradient_norm=float(np.linalg.norm(ga)),dimension=size,
    stencil_difference=float(np.linalg.norm(ga-gforward)/np.linalg.norm(ga)))
print('COMPRESS COMMON TRUST MODEL',flush=True)
model=HistoryModel(np.column_stack([Z,dc,dh.T,da]),np.column_stack([V@H,jc,jh.T,ja]),r,Z.shape[1]+1)
steps={}
for radius in radii:
    audit=greedy_history(model,7,radius);audit['radius']=radius;result['greedy'].append(audit)
    subsets={'H3':[4,5,6],'G2':audit['ladder'][1]['selected'],
             'G3':audit['ladder'][2]['selected'],'G4':audit['ladder'][3]['selected'],
             'G3+A':audit['ladder'][2]['selected']+[7]}
    _,pc,_=cauchy_step(r,dc,jc,radius,np.ones_like(x))
    for group,ids in subsets.items():
        s,pred,lam=model.full_step(ids,radius);rr=f(x+s);actual=float((r@r-rr@rr)/2)
        checks=[]
        for h in scales:
            js=derivative(s,h);p=float(-r@js-.5*(js@js))
            checks.append(dict(h=h,prediction=p,discrepancy=abs(p-pred)/max(abs(pred),1e-100)))
        raw=bool(pred>=pc*(1-1e-6) and np.linalg.norm(s)<=radius*(1+1e-6))
        feasible=bool(f.feasible(x+s))
        passed=bool(raw and feasible and actual>0 and pred>0 and actual/pred>.1 and
                    all(c['prediction']>0 and c['discrepancy']<.05 and actual/c['prediction']>.1 for c in checks))
        row=dict(group=group,radius=radius,selected=ids,prediction=pred,actual=actual,
            rho=actual/pred,residual=float(np.linalg.norm(rr)),step_norm=float(np.linalg.norm(s)),
            cauchy_prediction=pc,raw_model_pass=raw,feasible=feasible,passed=passed,checks=checks,lambda_value=lam)
        result['trials'].append(row);steps[group+'_'+str(radius)]=s
        print('TRIAL',group,radius,'indices',ids,'actual',actual,'rho',row['rho'],'pass',passed,flush=True);save()
result['seconds']=time.perf_counter()-start
np.savez_compressed(ROOT/'active_vectors.npz',x=x,residual=r,history=dh,history_responses=jh,core=dc,annulus=da,**steps)
save();print('DONE',result['seconds'],flush=True)
