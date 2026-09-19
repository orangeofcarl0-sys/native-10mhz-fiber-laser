"""Independent GPU batch/serial, energy derivative and wider-window checks."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_energy import EnergyResidual
from steady_window import embed
state=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
last=np.load(ROOT/'energy_09.npz');x=last['x']
rfun=EnergyResidual(state,.9,gpu=True);r=rfun(x)
other=x.copy();other[-1]+=1e-3
batch=rfun.batch(np.array([x,other]));serial=np.array([rfun(x),rfun(other)])
d=np.zeros_like(x);d[:4*rfun.n]=x[:4*rfun.n];d/=np.linalg.norm(d)
derivatives=[]
for h in [1e-5,5e-6,1e-6]:derivatives.append(float((rfun(x+h*d)[-1]-rfun(x-h*d)[-1])/(2*h)))
wide=dict(state)
for key in ['original_template','gauge_template','time_tangent']:
    wide[key]=np.pad(state[key],((0,0),(rfun.n//2,)*2))
wide['x']=embed(state['x'],rfun.n)
w=EnergyResidual(wide,.9,gpu=True);rw=w(embed(x,rfun.n));er=embed(r,rfun.n)
result=dict(batch_serial_relative=float(np.linalg.norm(batch-serial)/np.linalg.norm(serial)),
    energy_directional_derivatives=derivatives,energy_derivative_halving_relative=float(abs(derivatives[0]-derivatives[1])/max(abs(derivatives[1]),1e-100)),
    physical_residual=float(np.linalg.norm(r[:-1])),energy_relative_error=float(r[-1]),
    wide_physical_residual=float(np.linalg.norm(rw[:-1])),wide_energy_relative_error=float(rw[-1]),
    full_vector_relative_change=float(np.linalg.norm(rw-er)/np.linalg.norm(r)),
    root_threshold_met=bool(np.linalg.norm(r)<1e-7),certified_stable=False)
(ROOT/'energy_endpoint_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
