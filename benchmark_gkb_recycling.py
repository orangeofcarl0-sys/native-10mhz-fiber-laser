"""Isolated timing repeat after all unit tests; retain first-pass timings."""
import hashlib,json,time
import numpy as np
from config import ROOT,PROJECT
from gpu_setup import cp
from steady_descent_sources import physical_residual
from discrete_adjoint import DiscreteAdjoint
from steady_gkb import gkb,svd_trust
from steady_gkb_recycling import orthogonal_union

a=json.loads((ROOT/'audit.json').read_text(encoding='utf8'))
s=np.load(ROOT/'candidate_vectors.npz');x=s['x'];r=s['r']
old=np.load(ROOT/'local_full_bases.npz')['Vold']
f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True);adj=DiscreteAdjoint(f)
def measure(call):
    cp.cuda.Stream.null.synchronize();t=time.perf_counter();value=call()
    cp.cuda.Stream.null.synchronize();return value,time.perf_counter()-t
def derivative(v):
    h=1e-5/np.linalg.norm(v);return (f(x+h*v)-f(x-h*v))/(2*h)
count=0
def transpose(v):
    global count
    count+=1
    if count%32==0:print('TIMING COLUMN',count,flush=True)
    return adj.value_and_vjp(x,v)[1]
adj.value_and_vjp(x)
(u,v,jv,B,ladder),fresh_seconds=measure(lambda:gkb(derivative,transpose,-r,192))
error=float(np.linalg.norm(B-s['B'])/np.linalg.norm(s['B']));assert error<1e-10
jold,response_seconds=measure(lambda:np.column_stack([derivative(d) for d in old.T]))
def merge(basis,response):
    q,jq,info,transform=orthogonal_union(basis,response)
    out,small=np.linalg.qr(np.column_stack([-r,jq]),mode='reduced')
    y,p,lam=svd_trust(small[:,1:],out.T@(-r),a['radius'])
    return q@y,p
(sr,pr),rtime=measure(lambda:merge(old,jold))
(sc,pc),ctime=measure(lambda:merge(np.column_stack([v[:,:64],old]),np.column_stack([jv[:,:64],jold])))
assert abs(pc/next(t['prediction'] for t in a['trials'] if t['name']=='K64+R64')-1)<1e-7
a['first_pass_timing']=a['timing'].copy()
a['benchmark_repeat']=dict(note='Dedicated timing repeat after unit tests completed; no simultaneous validation jobs.',
    B_relative_error=error,source_sha256=hashlib.sha256((PROJECT/'benchmark_gkb_recycling.py').read_bytes()).hexdigest())
a['timing'].update(fresh192=fresh_seconds,recompute_old_responses=response_seconds,recycled_solve=rtime,union_merge_solve=ctime)
warm=ladder[63]['seconds']+response_seconds+ctime
a['timing'].update(union_warm=warm,union_cold=warm+a['timing']['old_rebuild'])
for t in a['trials']:
    t['first_pass_cost_seconds']=t['cost_seconds']
    if t['name'].startswith('K') and '+' not in t['name']:
        t['first_pass_build_seconds']=t['build_seconds'];t['build_seconds']=ladder[t['k']-1]['seconds']
        t['cost_seconds']=t['build_seconds']+t['solve_seconds']
    elif t['name']=='R64':t['cost_seconds']=response_seconds+rtime
    else:t['cost_seconds']=warm
for k in [128,192]:
    fresh=next(t for t in a['trials'] if t['name']=='K'+str(k));merged=a['trials'][-1]
    coverage=pc/fresh['prediction'];ratio=warm/fresh['cost_seconds']
    a['recycling_comparisons'][str(k)]=dict(model_coverage=coverage,cost_ratio=ratio,
        actual_ratio=merged['actual']/fresh['actual'] if fresh['actual']>0 else None,
        supported=bool(coverage>.9 and ratio<=.8 and merged['passed']))
(ROOT/'audit.json').write_text(json.dumps(a,indent=2),encoding='utf8')
print('ISOLATED BENCHMARK',a['timing'],a['recycling_comparisons'],flush=True)
