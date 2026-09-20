"""Forty pure adaptive GKB outer steps from the latest physical endpoint."""
import hashlib,json,time,zipfile
import numpy as np
from config import ROOT,PROJECT
from discrete_adjoint import DiscreteAdjoint
from steady_descent_sources import physical_residual,BASE
from steady_gkb_outer import solve_gkb

begin=time.perf_counter()
paths=[PROJECT/'results/steady_adjoint_continuation_20260920/trajectory.npz',
       PROJECT/'results/steady_tail_20260919/tail_continuation.npz']
x0=np.load(paths[0])['states'][-1].copy()
f=physical_residual(np.load(paths[1]),gpu=True);adj=DiscreteAdjoint(f)
r0=f(x0);assert abs(np.linalg.norm(r0)-.001096712976543593)<1e-13
protocol=dict(initial_residual=float(np.linalg.norm(r0)),initial_radius=.00625,physical_base=BASE,
 maximum_outer=40,maximum_k=64,checkpoints=list(range(8,65,8)),
 saturation='Two consecutive 8-column marginal model gains in [0,0.01) at initial outer radius',
 retries='Reuse frozen current-state basis when radius shrinks; no state update until guards pass',
 root_threshold=1e-7,milestones=[1e-3,1e-4,1e-5,1e-7],
 excluded='history, root-Z, C-preconditioner, annulus, persisted seed, parameter changes',
 multiple_shooting_diagnostic='After >=30 accepted steps, last10 all k>=48 and saturated, median rho>0.5, last5 mean relative decrease<0.0005, R>=1e-4. Diagnostic only; no automatic formulation change.',
 input_sha256={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
with zipfile.ZipFile(ROOT/'executed_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PROJECT.glob('*.py')):z.write(p,p.name)
    z.write(PROJECT/'parameters/effective_parameters.json','parameters/effective_parameters.json')
states=[x0.copy()]
def blocks(r):return dict(field=float(np.linalg.norm(r[:4*f.n])),population=float(np.linalg.norm(r[4*f.n:-2])),gauge=float(np.linalg.norm(r[-2:])))
block_history=[blocks(r0)]
def observe(x,r,rows):
    row=rows[-1];row['residual_blocks_after']=blocks(r)
    if 'accepted_trial' in row:
        states.append(x.copy());block_history.append(blocks(r))
        t=row['trials'][row['accepted_trial']]
        print('PURE GKB STEP',len(rows),'R',np.linalg.norm(r),'k',row['k'],'rho',t['rho'],'saturated',row['saturated'],flush=True)
    np.savez_compressed(ROOT/'trajectory.npz',states=np.array(states))
    (ROOT/'progress.json').write_text(json.dumps(dict(history=rows,residual=float(np.linalg.norm(r)),blocks=block_history,seconds=time.perf_counter()-begin),indent=2),encoding='utf8')

x,rows,status=solve_gkb(f,lambda x,v:adj.value_and_vjp(x,v)[1],x0,observer=observe)
r=f(x);norms=[float(np.linalg.norm(r0))]+[s['trials'][s['accepted_trial']]['residual'] for s in rows if 'accepted_trial' in s]
accepted=[s for s in rows if 'accepted_trial' in s]
gains=1-np.array(norms[1:])/np.array(norms[:-1]);last5=float(np.mean(gains[-5:])) if len(gains)>=5 else None
milestones={str(t):next((i for i,v in enumerate(norms) if v<t),None) for t in protocol['milestones']}
ms_gate=bool(len(accepted)>=30 and all(s['k']>=48 and s['saturated'] for s in accepted[-10:])
    and np.median([s['trials'][s['accepted_trial']]['rho'] for s in accepted[-10:]])>.5
    and last5<.0005 and np.linalg.norm(r)>=1e-4)
result=dict(status=status,initial_residual=norms[0],residual=float(np.linalg.norm(r)),history=rows,
 accepted_steps=len(accepted),norms=norms,blocks=block_history,milestones=milestones,
 last_five_mean_relative_decrease=last5,multiple_shooting_diagnostic=ms_gate,
 seconds=time.perf_counter()-begin,numerical_root=bool(np.linalg.norm(r)<1e-7),certified_stable=False)
np.savez_compressed(ROOT/'final.npz',x=x,r=r,output=f.output)
(ROOT/'continuation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('PURE GKB COMPLETE',status,result['residual'],milestones,flush=True)
