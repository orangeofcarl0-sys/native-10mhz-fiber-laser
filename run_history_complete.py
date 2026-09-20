"""Extend the frozen active-history audit; no annulus sweep or continuation."""
import hashlib,json,time,zipfile
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual,BASE
from steady_support_lu import build_factored
from steady_hookstep import arnoldi
from steady_cauchy import cauchy_step
from steady_active_history import HistoryModel,greedy_history
from steady_history_complete import exhaustive_history,paired_novelty

start=time.perf_counter();radii=[.00156,.003125,.00625];scales=[1e-5,3e-6,1e-6]
source=PROJECT/'results/steady_active_history_20260920'
template_path=PROJECT/'results/steady_tail_20260919/tail_continuation.npz'
inputs=[source/'active_vectors.npz',source/'active_history.json',template_path]
with zipfile.ZipFile(ROOT/'executed_sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(PROJECT.glob('*.py')):archive.write(path,path.name)
    archive.write(PROJECT/'parameters/effective_parameters.json','parameters/effective_parameters.json')
old=json.loads(inputs[1].read_text());saved=np.load(inputs[0]);x=saved['x'];dh=saved['history'];da=saved['annulus']
f=physical_residual(np.load(template_path),gpu=True);r=f(x)
assert np.linalg.norm(r-saved['residual'])<1e-14
result=dict(input_sha256={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    initial_residual=float(np.linalg.norm(r)),physical_base=BASE,radii=radii,history_outer_steps=old['history_outer_steps'],
    directions={},greedy=[],exhaustive=[],trials=[],baseline_replay=[],
    contract='Frozen S3 endpoint; all seven histories, persisted annulus, exhaustive equal-cardinality audit. No continuation, parameter release or fresh annulus sweep.')
def save(): (ROOT/'history_complete.json').write_text(json.dumps(result,indent=2),encoding='utf8')
save()
def derivative(d,h=1e-5):
    eps=h/max(np.linalg.norm(d),1e-100)
    return (f(x+eps*d)-f(x-eps*d))/(2*eps)
def checked_direction(d):
    js=[derivative(d,h) for h in scales];j=js[0]
    return j,dict(slope=float(r@j),response_norm=float(np.linalg.norm(j)),relative_checks=[float(np.linalg.norm(v-j)/np.linalg.norm(j)) for v in js])
print('REBUILD SHARED C / ARNOLDI',flush=True);cache={}
pre,build=build_factored(f,x,r,'C',descent=cache)
dc=-cache['lift'](cache['gradient']);dc/=np.linalg.norm(dc)
result['core_replay_difference']=float(np.linalg.norm(dc-saved['core']))
assert result['core_replay_difference']<1e-10
jc,result['directions']['core']=checked_direction(dc)
H,Z,y,linear,V=arnoldi(derivative,r,pre,limit=240,tolerance=.008,return_basis=True)
result['common_basis']=dict(dimension=len(y),internal_error=linear,build=build)
jh=[]
for i,d in enumerate(dh):
    j,info=checked_direction(d);jh.append(j);result['directions'][str(i)]=info
jh=np.array(jh);result['history_response_replay_difference']=float(np.linalg.norm(jh-saved['history_responses']))
ja,result['directions']['annulus']=checked_direction(da)
print('COMPRESS AND NOVELTY',flush=True)
model=HistoryModel(np.column_stack([Z,dc,dh.T,da]),np.column_stack([V@H,jc,jh.T,ja]),r,Z.shape[1]+1)
result['novelty']={str(t):paired_novelty(model.state[:,:-1],model.response[:,:-1],model.state[:,-1],model.response[:,-1],t) for t in [1e-8,1e-10,1e-12]}
np.savez_compressed(ROOT/'compressed_model.npz',state=model.state,response=model.response,beta=model.beta,base_count=model.base_count)
steps={}
for radius in radii:
    audit=greedy_history(model,7,radius,maximum=7);audit['radius']=radius;result['greedy'].append(audit)
    exact=exhaustive_history(model,7,radius);exact['radius']=radius;result['exhaustive'].append(exact)
    for base in [v for v in old['trials'] if v['radius']==radius]:
        s,p,_=model.full_step(base['selected'],radius)
        error=float(np.linalg.norm(s-saved[base['group']+'_'+str(radius)])/np.linalg.norm(s))
        pe=abs(p-base['prediction'])/base['prediction']
        result['baseline_replay'].append(dict(group=base['group'],radius=radius,step_relative=error,prediction_relative=pe))
        assert error<1e-6 and pe<1e-6
    subsets={f'G{k}':audit['ladder'][k-1]['selected'] for k in [5,6,7]}
    subsets.update({f'G{k}+A':audit['ladder'][k-1]['selected']+[7] for k in [4,5,6,7]})
    subsets['E4']=exact['best_by_count'][4]['selected']
    _,pc,_=cauchy_step(r,dc,jc,radius,np.ones_like(x))
    for group,ids in subsets.items():
        s,pred,lam=model.full_step(ids,radius);rr=f(x+s);actual=float((r@r-rr@rr)/2)
        checks=[]
        for h in scales:
            js=derivative(s,h);p=float(-r@js-.5*(js@js))
            checks.append(dict(h=h,prediction=p,discrepancy=abs(p-pred)/max(abs(pred),1e-100)))
        raw=bool(pred>=pc*(1-1e-6) and np.linalg.norm(s)<=radius*(1+1e-6));feasible=bool(f.feasible(x+s))
        passed=bool(raw and feasible and actual>0 and pred>0 and actual/pred>.1 and all(c['prediction']>0 and c['discrepancy']<.05 and actual/c['prediction']>.1 for c in checks))
        row=dict(group=group,radius=radius,selected=ids,prediction=pred,actual=actual,rho=actual/pred,
            residual=float(np.linalg.norm(rr)),step_norm=float(np.linalg.norm(s)),cauchy_prediction=pc,
            raw_model_pass=raw,feasible=feasible,passed=passed,checks=checks,lambda_value=lam)
        result['trials'].append(row);steps[group+'_'+str(radius)]=s
        print('TRIAL',group,radius,'actual',actual,'rho',row['rho'],'pass',passed,flush=True);save()
result['seconds']=time.perf_counter()-start
np.savez_compressed(ROOT/'complete_vectors.npz',x=x,residual=r,**steps)
save();print('DONE',result['seconds'],flush=True)
