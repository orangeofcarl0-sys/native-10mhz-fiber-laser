"""Independent central Jv check on the last saved continuation direction."""
import json
import numpy as np
from config import ROOT,config
from steady_window import restore

state=np.load(ROOT/'tail_continuation.npz')
residual,_=restore(config('CNT_OC',.2),state,gpu=True)
saved=np.load(ROOT/'continuation_last_direction.npz');x,r,d=saved['x'],saved['r'],saved['d']
np.testing.assert_allclose(residual(x),r,atol=1e-12)
length=np.linalg.norm(d);unit=d/length
central=(residual(x+1e-6*unit)-residual(x-1e-6*unit))/(2e-6)
forward=(residual(x+1e-7*unit)-r)/1e-7
result=dict(step=int(saved['step']),direction_norm=float(length),
    forward_central_relative_difference=float(np.linalg.norm(forward-central)/np.linalg.norm(central)),
    central_linear_relative_residual=float(np.linalg.norm(r+length*central)/np.linalg.norm(r)),
    forward_linear_relative_residual=float(np.linalg.norm(r+length*forward)/np.linalg.norm(r)))
(ROOT/'late_derivative_check.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result),flush=True)
