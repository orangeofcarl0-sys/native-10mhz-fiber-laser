"""Sequential matched20-step fresh192 vs fresh64+old64 experiment."""
import hashlib,json,os,time,zipfile
from pathlib import Path
import numpy as np
from config import ROOT,PROJECT
from discrete_adjoint import DiscreteAdjoint
from steady_descent_sources import physical_residual,BASE
from steady_gkb_globalized import solve_outer

source=Path(os.environ['LASER_BASIS_DIR']);basis_path=source/'local_full_bases.npz'
trajectory=PROJECT/'results/steady_pure_gkb_20260920/trajectory.npz'
x0=np.load(trajectory)['states'][-1].copy()
with np.load(basis_path) as a:old_basis=a['Vold'].copy()
protocol=dict(initial_residual=.0009746756121000674,initial_radius=.0125,accepted_budget=20,
 arms={'fresh':'fresh192 every state','recycled':'fresh64 + previous accepted state fresh64, current J responses'},
 physical_base=BASE,initial_recycled_cache='Vold from depth audit; pure40 states[-2], used to build accepted step40',
 shrink=.5,expand=1.5,shrink_rho=.25,expand_initial_rho=.8,expand_continue_rho=.5,
 acceptance='feasible, positive actual/prediction, rho>0.1, Cauchy, final3-scale Js5% gate; choose largest actual among tried eligible candidates',
 epsilonNL='diagnostic only',decision='both arms complete20 or root; P_B/P_A>=.9 and algorithm_seconds_B/A<=.75',
 timing='sequential same GPU; sum outer seconds excludes observer serialization; warm cached old basis, cold initialization reference reported separately',
 input_sha256={str(trajectory.relative_to(PROJECT)):hashlib.sha256(trajectory.read_bytes()).hexdigest(),'local_full_bases.npz':hashlib.sha256(basis_path.read_bytes()).hexdigest()})
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
with zipfile.ZipFile(ROOT/'executed_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in PROJECT.glob('*.py'):z.write(p,p.name)
    z.write(PROJECT/'parameters/effective_parameters.json','parameters/effective_parameters.json')
results={}
for arm in ['fresh','recycled']:
    folder=ROOT/arm;folder.mkdir(exist_ok=True)
    f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True);adj=DiscreteAdjoint(f)
    r0=f(x0);assert abs(np.linalg.norm(r0)-protocol['initial_residual'])<1e-13
    adj.value_and_vjp(x0) # identical warm-up, excluded from measured outer
    states=[x0.copy()];calls=0;begin=time.perf_counter()
    def transpose(x,v):
        global calls
        calls+=1
        if calls%64==0:print('BUILD',arm,'JTV',calls,flush=True)
        return adj.value_and_vjp(x,v)[1]
    def observe(x,r,rows):
        row=rows[-1]
        row['blocks']=dict(field=float(np.linalg.norm(r[:4*f.n])),population=float(np.linalg.norm(r[4*f.n:-2])),gauge=float(np.linalg.norm(r[-2:])))
        if 'accepted_trial' in row:
            states.append(x.copy());t=row['trials'][row['accepted_trial']]
            print('ACCEPT',arm,len(states)-1,'R',np.linalg.norm(r),'radius',t['radius'],'rho',t['rho'],'probes',len(row['trials']),flush=True)
        np.savez_compressed(folder/'trajectory.npz',states=np.array(states))
        (folder/'progress.json').write_text(json.dumps(dict(history=rows,residual=float(np.linalg.norm(r)),accepted=len(states)-1),indent=2),encoding='utf8')
    x,rows,status=solve_outer(f,transpose,x0,arm,old_basis=old_basis.copy(),observer=observe)
    r=f(x);norm=float(np.linalg.norm(r));accepted=sum('accepted_trial' in row for row in rows)
    result=dict(arm=arm,status=status,initial_residual=float(np.linalg.norm(r0)),residual=norm,
      accepted_steps=accepted,progress=float(-np.log(norm/np.linalg.norm(r0))),history=rows,
      algorithm_seconds=sum(row['seconds'] for row in rows),basis_seconds=sum(row['basis_seconds'] for row in rows),
      probe_seconds=sum(row['probe_seconds'] for row in rows),gate_seconds=sum(row['gate_seconds'] for row in rows),
      nonlinear_probes=sum(row['nonlinear_probes'] for row in rows),jtv_calls=calls,wall_seconds=time.perf_counter()-begin,
      numerical_root=bool(norm<1e-7),certified_stable=False)
    (folder/'continuation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    np.savez_compressed(folder/'final.npz',x=x,r=r)
    results[arm]=result
    print('ARM COMPLETE',arm,norm,result['progress'],result['algorithm_seconds'],flush=True)
a=results['fresh'];b=results['recycled'];p_ratio=b['progress']/a['progress'];t_ratio=b['algorithm_seconds']/a['algorithm_seconds']
summary=dict(arms={key:{k:v for k,v in value.items() if k!='history'} for key,value in results.items()},
 progress_ratio=p_ratio,time_ratio=t_ratio,basis_time_ratio=b['basis_seconds']/a['basis_seconds'],
 supports_recycled_default=bool(all(t['accepted_steps']==20 or t['numerical_root'] for t in [a,b]) and p_ratio>=.9 and t_ratio<=.75),
 cold_cache_rebuild_reference_seconds=json.loads((source/'audit.json').read_text(encoding='utf8'))['timing']['old_rebuild'])
(ROOT/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print('AB COMPLETE',summary,flush=True)
