"""Physical parameter derivatives with fixed state/gauge scales.

CNT saturation power means E_sat/tau; tau is held fixed. GDD changes the
SMF/NDF allocation while preserving the 20.42 m total length and EDF length.
"""
import numpy as np
from config import config
from steady_state import CavityResidual

PARAMETERS={'pump':(.05,.005,'W'),'gdd':(.2,.02,'ps^2'),
            'oc':(.8,.05,'fraction'),'psat':(40.,4.,'W')}


def parameter_residual(state,name,q=0.,gpu=False):
    base,scale,_=PARAMETERS[name];value=base+scale*q
    c=config('CNT_OC',value if name=='gdd' else .2)
    if name=='psat':c['sa_saturation_energy_pJ']=value*c['sa_recovery_ps']
    r=CavityResidual(c,float(state['dt']),state['original_template'],
                     value if name=='pump' else .05,value if name=='oc' else .8,gpu=gpu)
    r.scale=float(state['scale']);r.time_scale=float(state['time_scale'])
    r.template=state['gauge_template'].copy();r.time_tangent=state['time_tangent'].copy()
    return r


def parameter_column(state,name,x,h=1e-3,gpu=False):
    plus=parameter_residual(state,name,h,gpu);minus=parameter_residual(state,name,-h,gpu)
    counts_plus=[v[0] for v in plus.engine.ops];counts_minus=[v[0] for v in minus.engine.ops]
    if counts_plus!=counts_minus:
        raise ValueError('Parameter difference crosses a propagation mesh-count boundary')
    return (plus(x)-minus(x))/(2*h)
