"""Latest endpoint, forty guarded full-gradient/history outer iterations."""
import hashlib,json,time,zipfile
import numpy as np
from config import ROOT,PROJECT
from discrete_adjoint import DiscreteAdjoint
from steady_descent_sources import physical_residual,BASE
from steady_active_history import HistoryModel
from steady_seed_bank import prune_history
from steady_augmented_outer import solve_augmented
from steady_support_lu import build_factored

start=time.perf_counter()
folder=PROJECT/'results/steady_persisted_seed_20260920'
paths=[folder/'trajectory.npz',folder/'continuation.json',PROJECT/'results/steady_tail_20260919/tail_continuation.npz']
saved=np.load(paths[0]);prior=json.loads(paths[1].read_text());x0=saved['states'][-1].copy()
bank=[d.copy() for d in saved['bank']];ids=prior['final_pending_bank_ids'].copy()
assert len(bank)==len(ids)==12
radius=prior['history'][-1]['next_radius']
f=physical_residual(np.load(paths[2]),gpu=True);adj=DiscreteAdjoint(f)
initial=float(np.linalg.norm(f(x0)));assert abs(initial-.001130642702993049)<1e-14
states=[x0.copy()];directions=[];pending={};gradient_times=[]
protocol=dict(input_sha256={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
 physical_base=BASE,initial_residual=initial,initial_radius=radius,maximum_steps=40,capacity=12,
 initial_bank_ids=ids.copy(),strong_success='final R<1e-3',improved='last-five mean norm gain>0.002',
 sustained='last-five mean norm gain in [0.0005,0.002]',
 plateau='last-five mean norm gain<0.0002 AND mean full/history model gain<=1.1',
 excluded='No annulus, persisted seed, parameter release, period2, Floquet, multiple shooting or JTJ',
 gradient='General value_and_vjp each outer; every accepted full direction promoted next model',
 bank_rule='Same leave-one-out with current full direction included; chronological ties; capacity12 in every model')
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
with zipfile.ZipFile(ROOT/'executed_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PROJECT.glob('*.py')):z.write(p,p.name)
    z.write(PROJECT/'parameters/effective_parameters.json','parameters/effective_parameters.json')


def value_gradient(x):
    began=time.perf_counter();r,g=adj.value_and_vjp(x)
    gradient_times.append(time.perf_counter()-began)
    return r,g


def builder(f,x,r,cache):
    print('ADJOINT PRECONDITIONER',np.linalg.norm(r),flush=True)
    return build_factored(f,x,r,'C')  # Deliberately no descent cache/gradient.


def factory(x,r,d,jd,h,z,v,derivative,rebuilt,radius,row):
    global bank,ids
    count=len(bank);extra=np.column_stack(bank+[d])
    responses=np.column_stack([*[derivative(q) for q in bank],jd])
    model=HistoryModel(np.column_stack([z,extra]),np.column_stack([v@h,responses]),r,z.shape[1])
    selected,events=prune_history(model,count,12,count,radius)
    oldids=ids.copy()
    for event in events:event['removed_id']=oldids[event['removed']]
    bank=[bank[i] for i in selected];ids=[oldids[i] for i in selected]
    row.update(history_ids=ids.copy(),history_bank_size=len(bank),bank_pruning=events,
       full_history_count=sum(i.startswith('full_') for i in ids),
       old_history_count=sum(not i.startswith('full_') for i in ids),
       gradient_seconds=gradient_times[-1],
       residual_blocks=dict(field=float(np.linalg.norm(r[:4*f.n])),population=float(np.linalg.norm(r[4*f.n:-2])),gauge=float(np.linalg.norm(r[-2:]))))
    pending.update(direction=d.copy(),id=f"full_{row['step']+1}")
    def at_radius(rad):
        _,ph,_=model.solve(selected,rad)
        s,p,lam=model.full_step(selected+[count],rad)
        return s,dict(prediction=p,history_prediction=ph,gain=p/ph if ph>0 else None,lambda_value=lam)
    return at_radius


def observe(x,r,h):
    row=h[-1]
    for t in row['trials']:
        if 'seed_model' in t:t['gradient_model']=t.pop('seed_model')
    if 'accepted_trial' in row:
        t=row['trials'][row['accepted_trial']]
        states.append(x.copy());directions.append(pending['direction'].copy())
        bank.append(pending['direction']);ids.append(pending['id']);pending.clear()
        print('ADJOINT STEP',len(h),'R',np.linalg.norm(r),'gain',t['gradient_model']['gain'],
              'rho',t['rho'],'full_history',row['full_history_count'],flush=True)
    np.savez_compressed(ROOT/'checkpoint.npz',x=x,states=np.array(states),bank=np.array(bank),directions=np.array(directions))
    (ROOT/'progress.json').write_text(json.dumps(dict(history=h,residual=float(np.linalg.norm(r)),bank_ids=ids,seconds=time.perf_counter()-start),indent=2))


x,h,status=solve_augmented(f,x0,max_steps=40,radius=radius,builder=builder,observer=observe,
                           gate_policy='step',model_factory=factory,value_gradient=value_gradient)
r=f(x);np.savez_compressed(ROOT/'final.npz',x=x,output=f.output,r=r)
np.savez_compressed(ROOT/'trajectory.npz',states=np.array(states),directions=np.array(directions),bank=np.array(bank))
accepted=[s for s in h if 'accepted_trial' in s]
gains=[1-s['trials'][s['accepted_trial']]['residual']/s['residual'] for s in accepted]
mean=float(np.mean(gains[-5:])) if len(gains)>=5 else None
gm=[s['trials'][s['accepted_trial']]['gradient_model']['gain'] for s in accepted[-5:]]
model_mean=float(np.mean(gm)) if len(gm)==5 and all(v is not None for v in gm) else None
regime='insufficient_steps' if mean is None else ('improved' if mean>.002 else 'sustained' if mean>=.0005 else 'plateau' if mean<.0002 and model_mean is not None and model_mean<=1.1 else 'intermediate')
result=dict(status=status,initial_residual=initial,residual=float(np.linalg.norm(r)),accepted_steps=len(accepted),
 history=h,seconds=time.perf_counter()-start,last_five_mean_norm_gain=mean,last_five_mean_full_gain=model_mean,
 strong_success=bool(np.linalg.norm(r)<.001),regime=regime,final_pending_bank_ids=ids,
 residual_blocks=dict(field=float(np.linalg.norm(r[:4*f.n])),population=float(np.linalg.norm(r[4*f.n:-2])),gauge=float(np.linalg.norm(r[-2:]))),
 numerical_root=bool(np.linalg.norm(r)<1e-7),certified_stable=False)
(ROOT/'continuation.json').write_text(json.dumps(result,indent=2));print('COMPLETE',status,result['residual'],regime,flush=True)
