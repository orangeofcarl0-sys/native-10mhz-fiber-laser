"""Thirty accepted-state attempts with one immutable annulus seed; no reseeding."""
import hashlib,json,time,zipfile
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual,BASE
from steady_active_history import HistoryModel
from steady_history_complete import novelty
from steady_seed_bank import prune_history,trigger_diagnostic
from steady_augmented_outer import solve_augmented
from steady_support_lu import build_factored

start=time.perf_counter();source=PROJECT/'results/steady_active_history_20260920/active_vectors.npz'
template_path=PROJECT/'results/steady_tail_20260919/tail_continuation.npz'
saved=np.load(source);x0=saved['x'];seed=saved['annulus'].copy();seed_hash=hashlib.sha256(seed.tobytes()).hexdigest()
bank=[d.copy() for d in saved['history']];ids=[f'prior_{i}' for i in [1,7,11,14,17,18,28]]
states=[x0.copy()];fresh_vectors=[];fresh_states=[];fresh_ids=[];pending={}
f=physical_residual(np.load(template_path),gpu=True);initial=float(np.linalg.norm(f(x0)))
assert abs(initial-.0011883945817279438)<1e-14
with zipfile.ZipFile(ROOT/'executed_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PROJECT.glob('*.py')):z.write(p,p.name)
    z.write(PROJECT/'parameters/effective_parameters.json','parameters/effective_parameters.json')
protocol=dict(input_sha256={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,template_path]},
    physical_base=BASE,initial_radius=.003125,maximum_steps=30,history_capacity=12,seed_sha256=seed_hash,
    no_fresh_annulus_sweep=True,success_A='Final R<1e-3 OR last-five mean norm decrease>0.001',
    success_B='GA>1.15 at strictly more than half of completed fresh-preconditioner states; no reseeding',
    trigger='diagnostic only: GA<1.15 at 3 consecutive fresh states AND prior five accepted-step mean norm decrease<0.0002',
    bank_rule='Promote each accepted fresh C direction at next model; if over capacity, remove minimum leave-one-out loss with seed present at initial radius; oldest breaks numerical ties.')
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
def builder(f,x,r,cache):
    print('SEED BUILD',float(np.linalg.norm(r)),flush=True)
    return build_factored(f,x,r,'C',descent=cache)

def factory(x,r,d,jd,h,z,v,derivative,rebuilt,radius,row):
    global bank,ids
    count=len(bank);extra=np.column_stack(bank+[seed])
    responses=np.column_stack([derivative(column) for column in extra.T])
    model=HistoryModel(np.column_stack([z,d,extra]),np.column_stack([v@h,jd,responses]),r,z.shape[1]+1)
    selected,events=prune_history(model,count,12,count,radius)
    original_ids=ids.copy()
    for event in events:event['removed_id']=original_ids[event['removed']]
    bank=[bank[i] for i in selected];ids=[original_ids[i] for i in selected]
    history_columns=list(range(model.base_count))+[model.base_count+i for i in selected]
    nx,_=novelty(model.state[:,history_columns],model.state[:,-1])
    nr,_=novelty(model.response[:,history_columns],model.response[:,-1])
    row.update(history_bank_size=len(bank),history_ids=ids.copy(),bank_pruning=events,
        seed_age=row['step'],seed_novelty_state=nx['relative'],seed_novelty_response=nr['relative'],
        seed_response_norm=float(np.linalg.norm(responses[:,-1])),seed_slope=float(r@responses[:,-1]))
    pending.clear()
    if rebuilt:pending.update(direction=d.copy(),state=x.copy(),id=f"run_{row['step']+1}")
    def at_radius(rad):
        _,ph,_=model.solve(selected,rad)
        step,pa,lam=model.full_step(selected+[count],rad)
        return step,dict(prediction=pa,history_prediction=ph,gain=pa/ph if ph>0 else None,lambda_value=lam)
    return at_radius

def observe(x,r,h):
    row=h[-1]
    if 'accepted_trial' in row:
        trial=row['trials'][row['accepted_trial']]
        if len(h)==1:
            prior=json.loads((PROJECT/'results/steady_history_complete_20260920/history_complete.json').read_text())
            expected=next(v for v in prior['trials'] if v['group']=='G7+A' and v['radius']==.003125)
            assert abs(trial['residual']-expected['residual'])<1e-12
        row['trigger_diagnostic']=trigger_diagnostic(h)
        states.append(x.copy())
        if pending:
            bank.append(pending['direction']);ids.append(pending['id'])
            fresh_vectors.append(pending['direction']);fresh_states.append(pending['state']);fresh_ids.append(pending['id']);pending.clear()
        print('SEED STEP',len(h),'R',float(np.linalg.norm(r)),'GA',trial.get('seed_model',{}).get('gain'),
            'rho',trial['rho'],'bank',row['history_bank_size'],'fresh',row['precondition_rebuilt'],flush=True)
    np.savez_compressed(ROOT/'continuation_checkpoint.npz',x=x,bank=np.array(bank),seed=seed)
    (ROOT/'continuation_progress.json').write_text(json.dumps(dict(history=h,residual=float(np.linalg.norm(r)),seconds=time.perf_counter()-start),indent=2))

x,h,status=solve_augmented(f,x0,max_steps=30,radius=.003125,builder=builder,observer=observe,
    gate_policy='step',model_factory=factory)
r=f(x);np.savez_compressed(ROOT/'continuation_final.npz',x=x,output=f.output)
assert hashlib.sha256(seed.tobytes()).hexdigest()==seed_hash
np.savez_compressed(ROOT/'trajectory.npz',states=np.array(states),seed=seed,fresh_directions=np.array(fresh_vectors),fresh_states=np.array(fresh_states),bank=np.array(bank))
accepted=[row for row in h if 'accepted_trial' in row]
gains=[1-row['trials'][row['accepted_trial']]['residual']/row['residual'] for row in accepted]
fresh=[row for row in accepted if row['precondition_rebuilt']]
good=sum(row['trials'][row['accepted_trial']].get('seed_model',{}).get('gain',0)>1.15 for row in fresh)
mean=float(np.mean(gains[-5:])) if len(gains)>=5 else None
result=dict(status=status,initial_residual=initial,residual=float(np.linalg.norm(r)),accepted_steps=len(accepted),
    history=h,seconds=time.perf_counter()-start,fresh_states=len(fresh),fresh_seed_gain_above_115=good,
    last_five_mean_norm_gain=mean,success_A=bool(np.linalg.norm(r)<.001 or (mean is not None and mean>.001)),
    success_B=bool(good>len(fresh)/2),hypothetical_trigger_states=[row['step'] for row in accepted if row['trigger_diagnostic']['hypothetical_trigger']],
    actual_reseed_count=0,seed_sha256=seed_hash,fresh_ids=fresh_ids,final_pending_bank_ids=ids,
    numerical_root=bool(np.linalg.norm(r)<1e-7),certified_stable=False)
(ROOT/'continuation.json').write_text(json.dumps(result,indent=2));print('SEED RESULT',status,result['residual'],flush=True)
