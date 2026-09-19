"""Re-evaluate every endpoint and widen time window without changing state units."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_parameters import parameter_residual
from steady_window import embed
state=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
summary=json.loads((ROOT/'deep_summary.json').read_text());checks=[]
for row in summary:
    last=np.load(ROOT/(row['name']+'.npz'));x=last['x'];q=(row['pump_W']-.05)/.005
    rfun=parameter_residual(state,'pump',q,gpu=True);r=rfun(x)
    wide=dict(state)
    for key in ['original_template','gauge_template','time_tangent']:
        wide[key]=np.pad(state[key],((0,0),(rfun.n//2,)*2))
    rw=parameter_residual(wide,'pump',q,gpu=True)(embed(x,rfun.n))
    entry=dict(name=row['name'],physical_residual=float(np.linalg.norm(r)),
        wide_residual=float(np.linalg.norm(rw)),full_vector_relative_change=float(np.linalg.norm(rw-embed(r,rfun.n))/np.linalg.norm(r)),
        recorded_difference=float(abs(np.linalg.norm(r)-row['physical_residual'])),numerical_root=bool(np.linalg.norm(r)<1e-7),certified_stable=False)
    checks.append(entry);print(json.dumps(entry),flush=True)
(ROOT/'endpoint_checks.json').write_text(json.dumps(checks,indent=2))
