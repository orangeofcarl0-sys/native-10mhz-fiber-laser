"""Lift current projected weak directions and measure full central Jv responses."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_parameters import parameter_residual
from steady_window import embed,center
from steady_preconditioner import spectral_coordinates

state=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
residual=parameter_residual(state,'pump',gpu=True);x=state['x'];n=residual.n//2
restrict,lift,_=spectral_coordinates(n,residual.cells,384)
vectors=np.load(ROOT/'parameter_geometry_modes.npz')['right']
result=json.loads((ROOT/'parameter_geometry.json').read_text())
for row,v in zip(result['modes'],vectors[::-1]):
    d=embed(lift(v),n);jv=(residual(x+1e-6*d)-residual(x-1e-6*d))/(2e-6)
    row['full_Jv_norm']=float(np.linalg.norm(jv))
    projected=restrict(center(jv,n))
    row['central_projected_Jv_norm']=float(np.linalg.norm(projected))
    row['outside_projected_response_norm']=float(np.linalg.norm(jv-embed(lift(projected),n)))
    row['full_to_projected_ratio']=float(np.linalg.norm(jv)/row['sigma'])
(ROOT/'parameter_geometry.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result['modes'][:3]),flush=True)
