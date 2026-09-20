"""Independent forward/reduced-model replay and recorded primitive FD scales."""
import json
import numpy as np
from config import ROOT,PROJECT,config
from adjoint_primitives import kerr_value,kerr_vjp,cnt_value,cnt_vjp
from discrete_adjoint import DiscreteAdjoint
from steady_state import CavityResidual
from steady_descent_sources import physical_residual
from steady_hookstep import hookstep
from test_discrete_adjoint import dot_scan,SCALES

rng=np.random.default_rng(903)
a=rng.normal(size=(2,64))+1j*rng.normal(size=(2,64))
u=rng.normal(size=a.shape)+1j*rng.normal(size=a.shape)
v=rng.normal(size=a.shape)+1j*rng.normal(size=a.shape)
c=config('CNT_OC',.2)
primitives={
 'Kerr':dot_scan(lambda a:kerr_value(a,.17),lambda a,v:kerr_vjp(a,v,.17),a,u,v),
 'CNT':dot_scan(lambda a:cnt_value(a,c,.125)[0],lambda a,v:cnt_vjp(a,v,c,.125),a,u,v)}
mock=CavityResidual(c,.125,a,.0275,.8);adj=DiscreteAdjoint(mock);n=mock.n
field=np.fft.fft(a);x=np.r_[field.real.ravel(),field.imag.ravel(),np.full(mock.cells,.3)]
def unpack(x):return (x[:2*n]+1j*x[2*n:4*n]).reshape(2,n),x[4*n:]
def forward(x):
    field,pop=unpack(x);b,neq,_=adj.segment(field,pop,1)
    return np.r_[b.real.ravel(),b.imag.ravel(),neq]
def reverse(x,v):
    field,pop=unpack(x);vb,vn=unpack(v)
    a,p=adj.segment_vjp(field,pop,1,vb,vn)
    return np.r_[a.real.ravel(),a.imag.ravel(),p]
for kind in ['EDF full','EDF population output','EDF population input']:
    u=rng.normal(size=len(x));v=rng.normal(size=len(x))
    if kind.endswith('output'):v[:4*n]=0
    if kind.endswith('input'):u[:4*n]=0
    u/=np.linalg.norm(u);v/=np.linalg.norm(v)
    primitives[kind]=dot_scan(forward,reverse,x,u,v)
assert all(min(e)<1e-5 for e in primitives.values())

data=json.loads((ROOT/'adjoint_frozen.json').read_text())
assert [s['step'] for s in data['states']]==[5,11,20,27]
f=physical_residual(np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz'),gpu=True)
checks=[]
for row in data['states']:
    z=np.load(ROOT/f"frozen_{row['step']}.npz");m=np.load(ROOT/f"model_{row['step']}.npz")
    x=z['x'];r=f(x);base=int(m['base_count'])
    for t in row['trials']:
        s=z['step_'+t['name']];rr=f(x+s);actual=float((r@r-rr@rr)/2)
        ids=list(range(base))+{'history':[],'annulus':[base],'full':[base+1]}[t['name']]
        _,p,_=hookstep(m['response'][:,ids],m['state'][:,ids],float(m['beta']),row['radius'],np.ones(m['state'].shape[0]))
        assert abs(actual-t['actual'])<1e-16 and abs(p-t['prediction'])<1e-16
        checks.append(dict(step=row['step'],name=t['name'],actual_error=abs(actual-t['actual']),prediction_error=abs(p-t['prediction'])))
from gpu_setup import cp
device=cp.cuda.runtime.getDeviceProperties(0)
name=device['name'].decode() if isinstance(device['name'],bytes) else device['name']
result=dict(primitives=dict(scales=SCALES,relative_errors=primitives),candidate_replays=checks,
            hardware=dict(gpu=name,total_memory_bytes=int(device['totalGlobalMem']),cupy=cp.__version__,numpy=np.__version__))
(ROOT/'independent_checks.json').write_text(json.dumps(result,indent=2))
print('Independent candidates',len(checks),'primitive best',{k:min(v) for k,v in primitives.items()})
