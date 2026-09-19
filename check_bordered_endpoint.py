"""Re-evaluate the final physical parameter and a doubled time window."""
import json
import numpy as np
from config import ROOT
from steady_parameters import parameter_residual
from steady_window import embed,center

run=json.loads((ROOT/'bordered_pilot.json').read_text())
state=dict(np.load(ROOT/'bordered_pilot.npz'));name,q=run['name'],run['q']
residual=parameter_residual(state,name,q,gpu=True);x=state['x'];r=residual(x);n=residual.n
wide_state=dict(state)
for key in ['a','original_template','gauge_template','time_tangent']:
    wide_state[key]=np.pad(state[key],((0,0),(n//2,n//2)))
wide_state['x']=embed(x,n)
wide=parameter_residual(wide_state,name,q,gpu=True);rw=wide(wide_state['x'])
result=dict(physical_residual=float(np.linalg.norm(r)),wide_residual=float(np.linalg.norm(rw)),
    full_vector_relative_change=float(np.linalg.norm(rw-embed(r,n))/np.linalg.norm(r)),
    outer_residual_norm=float(np.linalg.norm(rw-embed(center(rw,n),n))),
    parameter_bound_margin_scaled=float(2-abs(q)),population_min=float(state['pop'].min()),population_max=float(state['pop'].max()),
    field_block_fraction=float(np.dot(r[:4*n],r[:4*n])/np.dot(r,r)))
np.testing.assert_allclose(result['physical_residual'],run['physical_residual'],rtol=1e-12)
(ROOT/'bordered_endpoint_check.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result),flush=True)
