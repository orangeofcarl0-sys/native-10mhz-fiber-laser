"""Independent replay of best gradient/Hookstep probes and residual blocks."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_parameters import parameter_residual
state=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz');rows=[]
for name in ['down_270','down_275']:
    record=json.loads((ROOT/(name+'.json')).read_text());directions=np.load(ROOT/(name+'_directions.npz'))
    x=np.load(PROJECT/record['input'])['x'];f=parameter_residual(state,'pump',(record['pump_W']-.05)/.005,gpu=True)
    g=min(record['gradient_line'],key=lambda v:v['residual']);k=min(range(6),key=lambda j:record['radius_sweep'][j]['residual'])
    for label,step,expected in [('initial',np.zeros_like(x),record['physical_residual']),
            ('gradient',g['step']*directions['gradient_direction'],g['residual']),
            ('hookstep',directions['hooksteps'][k],record['radius_sweep'][k]['residual'])]:
        r=f(x+step);squares=[float(np.dot(v,v)) for v in [r[:4*f.n],r[4*f.n:-2],r[-2:]]]
        row=dict(name=name,probe=label,residual=float(np.linalg.norm(r)),replay_difference=float(abs(np.linalg.norm(r)-expected)),
            field_squared=squares[0],population_squared=squares[1],gauge_squared=squares[2])
        assert row['replay_difference']<1e-12
        rows.append(row);print(json.dumps(row),flush=True)
(ROOT/'step_checks.json').write_text(json.dumps(rows,indent=2))
