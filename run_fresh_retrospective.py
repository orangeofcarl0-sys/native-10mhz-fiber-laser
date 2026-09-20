"""Four frozen trajectory states: replace only the persisted annulus direction."""
import hashlib,json,time,zipfile
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual,annulus_coordinates,streamed_gradient,BASE
from steady_support_lu import build_factored
from steady_hookstep import arnoldi
from steady_active_history import HistoryModel
from steady_cauchy import cauchy_step

start=time.perf_counter();selected_steps=[5,11,20,27]
folder=PROJECT/'results/steady_persisted_seed_20260920'
paths=[folder/'continuation.json',folder/'trajectory.npz',PROJECT/'results/steady_active_history_20260920/active_vectors.npz',PROJECT/'results/steady_tail_20260919/tail_continuation.npz']
previous=json.loads(paths[0].read_text());trajectory=np.load(paths[1]);original=np.load(paths[2]);template=np.load(paths[3])
bank={f'prior_{i}':d for i,d in zip([1,7,11,14,17,18,28],original['history'])}
bank.update(dict(zip(previous['fresh_ids'],trajectory['fresh_directions'])))
old=trajectory['seed'];f=physical_residual(template,gpu=True)
with zipfile.ZipFile(ROOT/'executed_sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(PROJECT.glob('*.py')):archive.write(p,p.name)
    archive.write(PROJECT/'parameters/effective_parameters.json','parameters/effective_parameters.json')
result=dict(steps=selected_steps,index_convention='Zero-based saved states; before report accepted steps 6,12,21,28',
    physical_base=BASE,input_sha256={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    protocol='Same state, history IDs, fresh C, Arnoldi basis and recorded accepted radius. Old replay must pass before fresh sweep. No continuation or trigger change.',
    decision_rule='At least 3/4 with Gold<1.1 and Gfresh>1.5; also require valid full-map fresh candidates for confirmed recovery.',states=[])
def save(): (ROOT/'fresh_retrospective.json').write_text(json.dumps(result,indent=2))
save()
for index in selected_steps:
    began=time.perf_counter();row=previous['history'][index];accepted=row['trials'][row['accepted_trial']]
    assert row['step']==index and row['precondition_rebuilt']
    x=trajectory['states'][index];r=f(x);assert abs(np.linalg.norm(r)-row['residual'])<1e-14
    radius=accepted['radius'];assert radius==row['radius']
    def derivative(d,h=1e-5):
        eps=h/max(np.linalg.norm(d),1e-100)
        return (f(x+eps*d)-f(x-eps*d))/(2*eps)
    print('STATE',index,'C BUILD',flush=True);cache={}
    pre,build=build_factored(f,x,r,'C',descent=cache)
    dc=-cache['lift'](cache['gradient']);dc/=np.linalg.norm(dc)
    np.testing.assert_allclose(dc,bank[f'run_{index+1}'],rtol=1e-9,atol=1e-11)
    jc=derivative(dc);H,Z,y,linear,V=arnoldi(derivative,r,pre,limit=240,tolerance=.008,return_basis=True)
    dh=np.column_stack([bank[key] for key in row['history_ids']]);jh=np.column_stack([derivative(d) for d in dh.T]);jo=derivative(old)
    base_states=np.column_stack([Z,dc,dh]);base_responses=np.column_stack([V@H,jc,jh])
    replay=HistoryModel(np.column_stack([base_states,old]),np.column_stack([base_responses,jo]),r,base_states.shape[1])
    old_step,old_prediction,_=replay.full_step([0],radius);_,history_prediction,_=replay.solve([],radius)
    step_error=float(np.linalg.norm(old_step-(trajectory['states'][index+1]-x))/np.linalg.norm(old_step))
    prediction_error=abs(old_prediction-accepted['seed_model']['prediction'])/old_prediction
    assert step_error<1e-6 and prediction_error<1e-6
    assert abs(history_prediction-accepted['seed_model']['history_prediction'])/history_prediction<1e-6
    print('STATE',index,'OLD REPLAY',step_error,'FRESH ANNULUS',flush=True)
    _,lift,size=annulus_coordinates(f.n,f.cells)
    gradient,forward=streamed_gradient(f,x,r,lift,size)
    fresh=-lift(gradient);fresh/=np.linalg.norm(fresh);jf=derivative(fresh)
    model=HistoryModel(np.column_stack([base_states,old,fresh]),np.column_stack([base_responses,jo,jf]),r,base_states.shape[1])
    _,mh,_=model.solve([],radius)
    entry=dict(step=index,report_step=index+1,radius=radius,residual=float(np.linalg.norm(r)),history_ids=row['history_ids'],
        core_rebuild=build,arnoldi_dimension=len(y),linear_error=linear,
        replay_step_error=step_error,replay_prediction_error=prediction_error,
        cos_x=float(old@fresh),cos_J=float(jo@jf/(np.linalg.norm(jo)*np.linalg.norm(jf))),
        fresh_descent=float(-r@jf),old_descent=float(-r@jo),gradient_norm=float(np.linalg.norm(gradient)),
        gradient_stencil_difference=float(np.linalg.norm(gradient-forward)/np.linalg.norm(gradient)),history_prediction=mh,trials=[])
    _,pc,_=cauchy_step(r,dc,jc,radius,np.ones_like(x));vectors={}
    for group,ids in [('history',[]),('old',[0]),('fresh',[1])]:
        s,p,lam=model.full_step(ids,radius);rr=f(x+s);actual=float((r@r-rr@rr)/2);checks=[]
        for h in [1e-5,3e-6,1e-6]:
            js=derivative(s,h);q=float(-r@js-.5*(js@js))
            checks.append(dict(h=h,prediction=q,relative_difference=abs(q-p)/max(abs(p),1e-100)))
        passed=bool(p>=pc*(1-1e-6) and np.linalg.norm(s)<=radius*(1+1e-6) and f.feasible(x+s)
            and actual>0 and actual/p>.1 and all(c['prediction']>0 and c['relative_difference']<.05 and actual/c['prediction']>.1 for c in checks))
        entry['trials'].append(dict(group=group,prediction=p,gain=p/mh,actual=actual,rho=actual/p,
            residual=float(np.linalg.norm(rr)),step_norm=float(np.linalg.norm(s)),passed=passed,checks=checks))
        vectors['step_'+group]=s
        print('STATE',index,group,'gain',p/mh,'actual',actual,'rho',actual/p,'pass',passed,flush=True)
    entry['Gold']=entry['trials'][1]['gain'];entry['Gfresh']=entry['trials'][2]['gain']
    entry['threshold_met']=bool(entry['Gold']<1.1 and entry['Gfresh']>1.5)
    entry['validated_recovery']=bool(entry['threshold_met'] and all(t['passed'] for t in entry['trials']))
    entry['seconds']=time.perf_counter()-began;result['states'].append(entry)
    np.savez_compressed(ROOT/f'state_{index}.npz',x=x,residual=r,old=old,fresh=fresh,jo=jo,jf=jf,**vectors)
    np.savez_compressed(ROOT/f'model_{index}.npz',state=model.state,response=model.response,beta=model.beta,base_count=model.base_count)
    save()
result.update(seconds=time.perf_counter()-start,threshold_count=sum(s['threshold_met'] for s in result['states']),
    validated_count=sum(s['validated_recovery'] for s in result['states']))
result['adjoint_decision_gate']=result['validated_count']>=3
save();print('COMPLETE',result['threshold_count'],result['validated_count'],flush=True)
