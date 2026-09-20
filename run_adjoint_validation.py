"""Gate general VJP on the four archived full-resolution states."""
import json,time
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual,annulus_coordinates
from discrete_adjoint import DiscreteAdjoint
from test_discrete_adjoint import SCALES

folder=PROJECT/'results/steady_fresh_retrospective_20260920'
records=json.loads((folder/'fresh_retrospective.json').read_text())
template=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
cpu=physical_residual(template,gpu=False);gpu=physical_residual(template,gpu=True)
ca,ga=DiscreteAdjoint(cpu),DiscreteAdjoint(gpu)
restrict,lift,size=annulus_coordinates(cpu.n,cpu.cells)
rng=np.random.default_rng(8841);results=[]
for row in records['states']:
    step=row['step'];archive=np.load(folder/f'state_{step}.npz');x=archive['x']
    began=time.perf_counter();rc,gc=ca.value_and_vjp(x);rg,gg=ga.value_and_vjp(x)
    r=gpu(x)
    parity=dict(forward=float(np.linalg.norm(rg-r)/np.linalg.norm(r)),
                cpu_forward=float(np.linalg.norm(rc-r)/np.linalg.norm(r)),
                vjp=float(np.linalg.norm(gc-gg)/np.linalg.norm(gc)))
    assert max(parity.values())<1e-9,parity
    stream=-restrict(archive['fresh'])*row['gradient_norm'];adj=restrict(gg)
    relative=float(np.linalg.norm(adj-stream)/np.linalg.norm(stream))
    cosine=float(adj@stream/(np.linalg.norm(adj)*np.linalg.norm(stream)))
    assert relative<1e-4 and cosine>1-1e-8
    d=-lift(adj)/np.linalg.norm(adj);h=1e-5
    jd=(gpu(x+h*d)-gpu(x-h*d))/(2*h)
    slope=float(-r@jd);alpha=min(row['radius'],slope/float(jd@jd))
    prediction=float(alpha*slope-.5*alpha**2*(jd@jd))
    js=archive['jf'];ss=float(-r@js);alphas=min(row['radius'],ss/float(js@js))
    ps=float(alphas*ss-.5*alphas**2*(js@js))
    assert abs(slope/row['fresh_descent']-1)<1e-4
    assert abs(alpha/alphas-1)<1e-4 and abs(prediction/ps-1)<1e-4
    dots=[]
    for kind in ['full','field_u','pop_u','phase_time_u','field_v','pop_v','gauge_v']:
        u=rng.normal(size=len(x));v=rng.normal(size=len(x))
        if kind=='field_u':u[4*gpu.n:]=0
        if kind=='pop_u':u[:4*gpu.n]=0;u[-2:]=0
        if kind=='phase_time_u':u[:-2]=0
        if kind=='field_v':v[4*gpu.n:]=0
        if kind=='pop_v':v[:4*gpu.n]=0;v[-2:]=0
        if kind=='gauge_v':v[:-2]=0
        u/=np.linalg.norm(u);v/=np.linalg.norm(v)
        right=float(u@gpu.vjp(x,v));checks=[]
        for h in SCALES:
            left=float(((gpu(x+h*u)-gpu(x-h*u))/(2*h))@v)
            checks.append(dict(h=h,left=left,right=right,absolute=abs(left-right),
                               relative=abs(left-right)/max(abs(left),abs(right),1e-14)))
        assert min(t['relative'] for t in checks)<1e-5,(step,kind,checks)
        # A general cotangent CPU/GPU comparison, not only J^T R.
        if kind=='full':
            vg=gpu.vjp(x,v);vc=cpu.vjp(x,v)
            parity['random_vjp']=float(np.linalg.norm(vg-vc)/np.linalg.norm(vc))
            assert parity['random_vjp']<1e-9
        dots.append(dict(kind=kind,scales=checks))
    results.append(dict(step=step,parity=parity,annulus_relative=relative,
                        annulus_cosine=cosine,annulus_norm=float(np.linalg.norm(adj)),
                        slope=slope,alpha_C=alpha,prediction_C=prediction,
                        streaming_alpha_C=alphas,streaming_prediction_C=ps,dots=dots,
                        seconds=time.perf_counter()-began))
    np.savez_compressed(ROOT/f'gradient_{step}.npz',x=x,r=r,gradient=gg,
                        cpu_gradient=gc,annulus=adj,streaming_annulus=stream)
    (ROOT/'adjoint_validation.json').write_text(json.dumps(dict(states=results),indent=2))
    print('VALIDATED',step,parity,'annulus',relative,flush=True)
