"""Four frozen models and synchronized GPU benchmarks; never continue a state."""
import hashlib,json,time,zipfile
import numpy as np
from config import ROOT,PROJECT
from gpu_setup import cp
from discrete_adjoint import DiscreteAdjoint
from steady_descent_sources import physical_residual,annulus_coordinates,streamed_gradient
from steady_support_lu import build_factored
from steady_hookstep import arnoldi
from steady_active_history import HistoryModel
from steady_cauchy import cauchy_step

folder=PROJECT/'results/steady_persisted_seed_20260920'
oldfolder=PROJECT/'results/steady_fresh_retrospective_20260920'
previous=json.loads((folder/'continuation.json').read_text())
oldresults=json.loads((oldfolder/'fresh_retrospective.json').read_text())
validation=json.loads((ROOT/'adjoint_validation.json').read_text())
assert [s['step'] for s in validation['states']]==[5,11,20,27]
trajectory=np.load(folder/'trajectory.npz')
original=np.load(PROJECT/'results/steady_active_history_20260920/active_vectors.npz')
template=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
f=physical_residual(template,gpu=True);adjoint=DiscreteAdjoint(f)
bank={f'prior_{i}':d for i,d in zip([1,7,11,14,17,18,28],original['history'])}
bank.update(dict(zip(previous['fresh_ids'],trajectory['fresh_directions'])))
with zipfile.ZipFile(ROOT/'executed_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PROJECT.glob('*.py')):z.write(p,p.name)
    z.write(PROJECT/'parameters/effective_parameters.json','parameters/effective_parameters.json')
inputs=[folder/'trajectory.npz',folder/'continuation.json',oldfolder/'fresh_retrospective.json',
        PROJECT/'results/steady_tail_20260919/tail_continuation.npz',
        PROJECT/'results/steady_active_history_20260920/active_vectors.npz']
inputs += [oldfolder/f'state_{s}.npz' for s in [5,11,20,27]]
result=dict(baseline='5d7230838f4d2fff7e7bf8ac67c44b4ce78a6ff4',
 input_sha256={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
 memory_definition='Peak reserved bytes of an isolated CuPy pool retained to call end; upper bound on live pool tensors, excludes CUDA context and pre-existing arrays/FFT plans. Not whole-process peak.',
 timing_definition='Synchronized wall clock, warm kernels/plans. Forward and value+VJP medians of five; one full 6144-column central streaming sweep per state. VJP includes primal residual.',states=[])


def measured(function,repeats):
    durations=[];reservations=[]
    for _ in range(repeats):
        pool=cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(pool.malloc):
            cp.cuda.Stream.null.synchronize();start=time.perf_counter()
            answer=function()
            cp.cuda.Stream.null.synchronize();durations.append(time.perf_counter()-start)
            reservations.append(pool.total_bytes())
        pool.free_all_blocks()
    return answer,dict(seconds=durations,median_seconds=float(np.median(durations)),
                       peak_reserved_bytes=max(reservations))


begin=time.perf_counter()
for oldrow in oldresults['states']:
    step=oldrow['step'];row=previous['history'][step];x=trajectory['states'][step]
    r=f(x);radius=oldrow['radius'];stored=np.load(oldfolder/f'state_{step}.npz')
    print('STATE',step,'BENCHMARK',flush=True)
    adjoint.value_and_vjp(x);f.batch(np.tile(x,(16,1)))  # warm up both batch sizes
    _,forward_time=measured(lambda:f(x),5)
    (_,gradient),adj_time=measured(lambda:adjoint.value_and_vjp(x),5)
    restrict,lift,size=annulus_coordinates(f.n,f.cells)
    (ga,gforward),stream_time=measured(lambda:streamed_gradient(f,x,r,lift,size),1)
    annulus_error=float(np.linalg.norm(restrict(gradient)-ga)/np.linalg.norm(ga))
    assert annulus_error<1e-4
    fresh=-lift(ga)/np.linalg.norm(ga)
    assert np.linalg.norm(fresh-stored['fresh'])<1e-7
    print('STATE',step,'CORE BUILD','speedup',stream_time['median_seconds']/adj_time['median_seconds'],flush=True)
    cache={};pre,build=build_factored(f,x,r,'C',descent=cache)
    core=cache['restrict'](gradient)
    core_error=float(np.linalg.norm(core-cache['gradient'])/np.linalg.norm(core))
    assert core_error<1e-4,core_error
    dc=-cache['lift'](cache['gradient']);dc/=np.linalg.norm(dc)
    np.testing.assert_allclose(dc,bank[f'run_{step+1}'],rtol=1e-9,atol=1e-11)
    def derivative(d,h=1e-5):
        eps=h/max(np.linalg.norm(d),1e-100)
        return (f(x+eps*d)-f(x-eps*d))/(2*eps)
    H,Z,y,linear,V=arnoldi(derivative,r,pre,limit=240,tolerance=.008,return_basis=True)
    dh=np.column_stack([bank[key] for key in row['history_ids']])
    base=np.column_stack([Z,dc,dh])
    response=np.column_stack([V@H,derivative(dc),*[derivative(d) for d in dh.T]])
    full=-gradient/np.linalg.norm(gradient);jf=derivative(fresh);jg=derivative(full)
    model=HistoryModel(np.column_stack([base,fresh,full]),np.column_stack([response,jf,jg]),r,base.shape[1])
    _,pc,_=cauchy_step(r,dc,derivative(dc),radius,np.ones_like(x))
    _,mh,_=model.solve([],radius)
    entry=dict(step=step,radius=radius,history_ids=row['history_ids'],
               forward=forward_time,adjoint=adj_time,streaming=stream_time,
               speedup=stream_time['median_seconds']/adj_time['median_seconds'],
               forward_cost=adj_time['median_seconds']/forward_time['median_seconds'],
               annulus_relative=annulus_error,core_relative=core_error,
               core_build=build,linear_error=linear,trials=[])
    vectors={}
    for name,ids in [('history',[]),('annulus',[0]),('full',[1])]:
        s,p,lam=model.full_step(ids,radius);rr=f(x+s)
        actual=float((r@r-rr@rr)/2);checks=[]
        for h in [1e-5,3e-6,1e-6]:
            js=derivative(s,h);q=float(-r@js-.5*(js@js))
            checks.append(dict(h=h,prediction=q,relative=abs(q-p)/max(abs(p),1e-100)))
        passed=bool(p>=pc*(1-1e-6) and np.linalg.norm(s)<=radius*(1+1e-6)
                    and f.feasible(x+s) and actual>0 and actual/p>.1
                    and all(c['prediction']>0 and c['relative']<.05 for c in checks))
        entry['trials'].append(dict(name=name,prediction=p,gain=p/mh,actual=actual,rho=actual/p,
                                   passed=passed,step_norm=float(np.linalg.norm(s)),checks=checks))
        vectors['step_'+name]=s
        if name in ['history','annulus']:
            prior=next(t for t in oldrow['trials'] if t['group']==('fresh' if name=='annulus' else 'history'))
            assert abs(p/prior['prediction']-1)<1e-6
            assert abs(actual/prior['actual']-1)<1e-5
        print('STATE',step,name,'gain',p/mh,'rho',actual/p,'pass',passed,flush=True)
    entry['full_to_annulus_prediction']=entry['trials'][2]['prediction']/entry['trials'][1]['prediction']
    entry['full_to_annulus_actual']=entry['trials'][2]['actual']/entry['trials'][1]['actual']
    result['states'].append(entry);result['seconds']=time.perf_counter()-begin
    np.savez_compressed(ROOT/f'frozen_{step}.npz',x=x,r=r,gradient=gradient,stream_annulus=ga,
                        core_adjoint=core,core_stream=cache['gradient'],**vectors)
    np.savez_compressed(ROOT/f'model_{step}.npz',state=model.state,response=model.response,beta=model.beta,base_count=model.base_count)
    (ROOT/'adjoint_frozen.json').write_text(json.dumps(result,indent=2))
print('FROZEN COMPLETE',flush=True)
