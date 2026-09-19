"""Small multi-direction trust models and streamed spectral-gradient probes."""
import numpy as np
from config import config
from steady_state import CavityResidual
from steady_preconditioner import spectral_coordinates
from steady_hookstep import hookstep

SCALES={'pump':.005,'gdd':.02,'oc':.05,'psat':4.}
BASE={'pump':.0275,'gdd':.2,'oc':.8,'psat':40.}


def physical_residual(template, changes=None, gpu=False):
    """Scaled parameter offsets about the CURRENT 27.5 mW state, fixed gauges."""
    values=BASE.copy()
    for name,q in (changes or {}).items():values[name]+=SCALES[name]*q
    c=config('CNT_OC',values['gdd'])
    c['sa_saturation_energy_pJ']=values['psat']*c['sa_recovery_ps']
    f=CavityResidual(c,float(template['dt']),template['original_template'],values['pump'],values['oc'],gpu=gpu)
    f.scale=float(template['scale']);f.time_scale=float(template['time_scale'])
    f.template=template['gauge_template'].copy();f.time_tangent=template['time_tangent'].copy()
    return f


def annulus_coordinates(n,cells,inner=768,outer=1536):
    """Real orthonormal optical coordinates only; no duplicated population/gauges."""
    indices=np.r_[np.arange(inner+1,outer+1),np.arange(n-outer,n-inner)]
    restrict0,lift0,size=spectral_coordinates(n,cells,outer,indices)
    tail=cells+2
    return lambda x:restrict0(x)[:-tail],lambda g:lift0(np.r_[g,np.zeros(tail)]),size-tail


def streamed_gradient(f,x,r,lift,size,h=1e-6):
    """Full-output central gradient; retain forward comparison, no dense matrix."""
    central=np.empty(size);forward=np.empty(size)
    for start in range(0,size,16):
        count=min(16,size-start);unit=np.zeros((count,size));unit[np.arange(count),start+np.arange(count)]=1
        dx=np.array([h*lift(v) for v in unit]);plus=f.batch(x+dx);minus=f.batch(x-dx)
        central[start:start+count]=((plus-minus)/(2*h))@r
        forward[start:start+count]=((plus-r)/h)@r
    return central,forward


def enriched_model(h,z,v,r,directions,responses):
    """Keep original Arnoldi responses; QR output space includes the exact target."""
    state_basis=np.column_stack([z,directions])
    output_basis=np.column_stack([v@h,responses])
    q,_=np.linalg.qr(np.column_stack([-r/np.linalg.norm(r),output_basis]),mode='reduced')
    if np.dot(q[:,0],-r)<0:q[:,0]*=-1
    return q.T@output_basis,state_basis


def guarded_step(model,r,radius,cauchy_prediction):
    h,z=model
    step,prediction,lam=hookstep(h,z,np.linalg.norm(r),radius,np.ones(z.shape[0]))
    passed=bool(prediction>=cauchy_prediction*(1-1e-6) and np.linalg.norm(step)<=radius*(1+1e-6))
    return step,dict(prediction=prediction,cauchy_prediction=cauchy_prediction,raw_model_pass=passed,lambda_value=lam)
