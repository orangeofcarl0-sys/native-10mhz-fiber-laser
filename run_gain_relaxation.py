"""Direct physical-time continuation of g08_c53; no gain acceleration.

First budget: 10000 additional accepted RT. Extend to 20000 if conditional
gain age is still below 7. Boundary retries do not advance time or gain age.
The first 600 RT have unknown rate history, so age starts at zero at restart.
"""

import os, json, time, hashlib
from pathlib import Path
from collections import deque
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks
from config import ROOT, PROJECT, config
from gpu_engine import BatchEngine
from batch_engine import BatchEngine as CPUReference
from gain_age import advance, summary
from adaptive_solver import health, regrid, remap_population
from attractor_search import field_distance


LIMITS = dict(field=1e-3, intensity=1e-3, stokes=1e-3, population=1e-4,
              population_abs=1e-5, energy=1e-4, cnt=1e-8)


def periods(fields, pops, qs):
    """Finite-record recurrence only; perturbation stability remains untested."""
    if len(fields) < 65:
        return []
    fields, pops, qs = list(fields), list(pops), list(qs)
    energy=np.array([np.sum(abs(a)**2) for a in fields])
    answer=[]
    for p in range(1,17):
        errors=np.array([field_distance(fields[k-p],fields[k]) for k in range(p,len(fields))])
        nr=max(np.linalg.norm(pops[k]-pops[k-p])/np.linalg.norm(pops[k-p]) for k in range(p,len(pops)))
        na=max(float(np.max(abs(pops[k]-pops[k-p]))) for k in range(p,len(pops)))
        row=dict(period=p,intensity=float(errors[:,0].max()),field=float(errors[:,1].max()),
                 stokes=float(errors[:,2].max()),population=float(nr),population_abs=na,
                 energy=float(np.max(abs(energy[p:]/energy[:-p]-1))),
                 cnt=float(np.max(abs(np.array(qs[p:])-np.array(qs[:-p])))))
        row['passes_screen']=all(row[k] <= value for k,value in LIMITS.items())
        answer.append(row)
    return answer


def grid_probe(c, a, pop, q, dt, rounds=8):
    """8-RT local discrepancy against dt/2 and both propagation steps/2.

    This is an error estimate, not a proof of long-time grid convergence.
    Acceptance tolerances are not loosened to accommodate a high error floor.
    """
    reference=BatchEngine(c,dt,a.shape[-1],[.05],[.8])
    fine_c=dict(c,max_step_m=c['max_step_m']/2,passive_step_m=c['passive_step_m']/2)
    b,fdt=regrid(a.copy(),dt,finer=True)
    cells=int(np.ceil(c['segments'][1,0]/fine_c['max_step_m']))
    pb=remap_population(pop,cells);qb=q.copy()
    fine=BatchEngine(fine_c,fdt,b.shape[-1],[.05],[.8])
    x,px,qx=a.copy(),pop.copy(),q.copy()
    for _ in range(rounds):
        x,_,px,qx,_=reference.step(x,px,qx)
        b,_,pb,qb,_=fine.step(b,pb,qb)
    from scipy.signal import resample
    bx=resample(b,x.shape[-1],axis=-1)
    ir,fr,sr=field_distance(x[0],bx[0])
    pcoarse=remap_population(pb,px.shape[-1])
    result=dict(rounds=rounds,intensity=ir,field=fr,stokes=sr,
                population=float(np.linalg.norm(px-pcoarse)/np.linalg.norm(px)),
                population_abs=float(np.max(abs(px-pcoarse))),
                energy=float(abs(np.sum(abs(b)**2)*fdt/(np.sum(abs(x)**2)*dt)-1)))
    result['below_screen_tolerances']=all(result[k] <= LIMITS[k] for k in result if k in LIMITS)
    return result


def run():
    source=Path(os.environ['LASER_SCAN_DIR'])/'fine_group_08.mat'
    checkpoint=ROOT/'checkpoint.npz'
    manifest_path=ROOT/'manifest.json'
    c=config('CNT_OC',.2)
    hashes={n:hashlib.sha256((PROJECT/n).read_bytes()).hexdigest() for n in ['spectral_engine.py','gpu_kernels.py','gpu_engine.py','gain_age.py','run_gain_relaxation.py','parameters/effective_parameters.json']}
    if checkpoint.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['code_sha256']!=hashes:raise RuntimeError('Resume requires unchanged code')
        if manifest['status']!='running':
            print('Run already finalized',flush=True);return
        with np.load(checkpoint) as m:
            a,pop,q,age=(m[k].copy() for k in ['a','pop','q','age'])
            dt=float(m['dt']);done=int(m['done'])
        trace=json.loads((ROOT/'trace.json').read_text())[:done]
    else:
        m=loadmat(source,variable_names=['a','pop','q','completed'])
        assert m['completed'].ravel()[53]==600
        a=m['a'][53:54].copy();pop=m['pop'][53:54].copy();q=m['q'].ravel()[53:54].copy()
        dt=.125;age=np.zeros_like(pop);done=0;trace=[]
        manifest=dict(status='running',case='g08_c53',start_round=600,
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),code_sha256=hashes,
                      age_origin='restart at RT600; earlier rate history unknown and not imputed',
                      memory_meaning='conditional frozen-rate homogeneous coefficient, not coupled sensitivity',
                      primary_budget=10000,maximum_budget=20000,gain_age_gate=7,
                      recurrence_screen_limits=LIMITS,events=[],period_checks=[],
                      started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
        # Check the new fused B diagnostic independently against the CPU engine.
        eng=BatchEngine(c,dt,a.shape[-1],[.05],[.8]);cpu=CPUReference(c,dt,a.shape[-1],[.05],[.8])
        ga=eng.step(a.copy(),pop.copy(),q.copy());ca=cpu.step(a.copy(),pop.copy(),q.copy())
        np.testing.assert_allclose(eng.rates_b,cpu.rates_b,rtol=1e-11)
        for actual,expected in zip(ga,ca):np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)
        manifest['gpu_rate_validation_max_relative']=float(np.max(abs(eng.rates_b/cpu.rates_b-1)))
        manifest['initial_grid_probe']=grid_probe(c,a,pop,q,dt)
        print('INITIAL VALIDATION',manifest['initial_grid_probe'],flush=True)
    engine=BatchEngine(c,dt,a.shape[-1],[.05],[.8])
    fields,pops,qs=(deque(maxlen=65) for _ in range(3))
    start=time.perf_counter();status='running';retries=0
    def save():
        with (ROOT/'checkpoint.tmp').open('wb') as file:
            np.savez(file,a=a,pop=pop,q=q,age=age,dt=dt,done=done)
        (ROOT/'checkpoint.tmp').replace(checkpoint)
        (ROOT/'trace.json').write_text(json.dumps(trace),encoding='utf8')
        manifest.update(status=status,accepted_additional_rounds=done,total_rounds=600+done,**summary(age))
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    save()
    while done < 20000:
        candidate,out,pnext,qnext,sa=engine.step(a.copy(),pop.copy(),q.copy())
        edges=np.maximum(health(candidate,dt),health(out,dt))
        if edges[0]>1e-6 or edges[1]>1e-8:
            if a.shape[-1]*2>65536 or retries>=8:
                status='numerical_boundary_unresolved';manifest['failed_edges']=edges.tolist();break
            wider=bool(edges[0]>1e-6);finer=not wider
            a,dt=regrid(a,dt,finer=finer,wider=wider)
            manifest['events'].append(dict(at_total_round=600+done,action='rollback_widen' if wider else 'rollback_finer_dt',dt_ps=dt,n=a.shape[-1]))
            engine=BatchEngine(c,dt,a.shape[-1],[.05],[.8]);fields.clear();pops.clear();qs.clear();retries+=1
            print('RETRY',manifest['events'][-1],flush=True);continue
        retries=0
        a,pop,q=candidate,pnext,qnext
        age=advance(age,engine.rates_b,1/engine.e.rep)
        done+=1
        power=np.sum(abs(out[0])**2,axis=0)
        peak_count=len(find_peaks(power,height=.1*power.max(),prominence=.1*power.max())[0])
        values=dict(round=600+done,energy_pJ=float(power.sum()*dt),cnt_peak_W=float(sa[0,1]),
                    cnt_q_min=float(sa[0,2]),inversion_gap=float(engine.last_gap[0]),
                    tau_max_us=float(engine.last_tau[0]*1e6),peaks=peak_count,
                    time_edge=float(edges[0]),spectral_edge=float(edges[1]),dt_ps=dt,n=a.shape[-1],**summary(age))
        trace.append(values)
        fields.append(a[0].copy());pops.append(pop[0].copy());qs.append(float(q[0]))
        if done%500==0:
            pr=periods(fields,pops,qs)
            manifest['period_checks'].append(dict(round=600+done,periods=pr))
            tail=np.array([r['energy_pJ'] for r in trace[-500:]])
            print(dict(round=600+done,age=values['gain_age_min'],energy_nJ=values['energy_pJ']/1000,
                       cv500=float(tail.std()/max(tail.mean(),1e-250)),peaks=peak_count,
                       best_field=min((v['field'] for v in pr),default=None),seconds=time.perf_counter()-start),flush=True)
            save()
        elif done%100==0:
            print(dict(round=600+done,age=values['gain_age_min'],energy_nJ=values['energy_pJ']/1000),flush=True)
        if done>=10000 and summary(age)['gain_age_min']>=7:
            status='time_budget_complete_gain_age_reached';break
    if status=='running':status='maximum_budget_reached'
    manifest['final_periods']=periods(fields,pops,qs)
    manifest['final_grid_probe']=grid_probe(c,a,pop,q,dt) if a.shape[-1]<=32768 else {'status':'resource_limit'}
    passed=[r for r in manifest['final_periods'] if r['passes_screen']]
    manifest['classification']='no_period_1_to_16_recurrence_detected'
    if passed:
        manifest['classification']='finite_record_recurrence_candidate_needs_long_hold_and_perturbation_tests'
        manifest['candidate_period']=passed[0]['period']
    if status=='numerical_boundary_unresolved':manifest['classification']=status
    manifest['certified_stable']=False
    manifest['finished_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
    save()
    print('FINISHED',status,done,summary(age),manifest['classification'],flush=True)


if __name__=='__main__':run()
