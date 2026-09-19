"""Nonzero field closure and doubled-window checks at the actual released pump."""
import json
import numpy as np
from config import ROOT
from steady_parameters import parameter_residual
from steady_window import embed,center

run=json.loads((ROOT/'bordered_expanded.json').read_text())
state=dict(np.load(ROOT/'bordered_expanded.npz'));q=run['q']
residual=parameter_residual(state,'pump',q,gpu=True);x=state['x'];r=residual(x);n=residual.n
wide_state=dict(state)
for key in ['a','original_template','gauge_template','time_tangent']:
    wide_state[key]=np.pad(state[key],((0,0),(n//2,n//2)))
wide_state['x']=embed(x,n)
wide=parameter_residual(wide_state,'pump',q,gpu=True);rw=wide(wide_state['x'])
result=dict(physical_residual=float(np.linalg.norm(r)),
    relative_field_residual=float(np.linalg.norm(r[:4*n])*residual.scale/np.linalg.norm(state['a'])),
    field_energy_nJ=float(np.sum(abs(state['a'])**2)*.125/1000),
    population_gap=float(np.max(abs(state['pop']-residual.neq))),
    wide_residual=float(np.linalg.norm(rw)),
    full_vector_relative_change=float(np.linalg.norm(rw-embed(r,n))/np.linalg.norm(r)),
    outer_residual_norm=float(np.linalg.norm(rw-embed(center(rw,n),n))),
    parameter_lower_margin_scaled=float(q+9),parameter_upper_margin_scaled=float(2-q))
np.testing.assert_allclose(result['physical_residual'],run['physical_residual'],rtol=1e-12)
(ROOT/'expanded_endpoint_check.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result),flush=True)
