"""Locate residual spectral support before selecting a coarse inverse space."""
import os,json
from pathlib import Path
import numpy as np
from config import ROOT,config
from steady_state import CavityResidual
saved=np.load(Path(os.environ['LASER_STEADY_DIR'])/'best_pilot_larger_krylov.npz')
residual=CavityResidual(config('CNT_OC',.2),.125,saved['a'],.05,.8,gpu=True)
x=np.load(ROOT/'linear_none.npz')['x'];r=residual(x);n=residual.n
field=np.fft.fft(saved['a']);rf=np.fft.fft((r[:2*n]+1j*r[2*n:4*n]).reshape(2,n))
sf=np.sum(abs(field)**2,axis=0);sr=np.sum(abs(rf)**2,axis=0)
score=sf/sf.sum()+sr/sr.sum();order=np.argsort(score)[::-1]
rows=[]
for count in [65,129,257,513]:
    ix=order[:count]
    rows.append(dict(bins=count,field_fraction=float(sf[ix].sum()/sf.sum()),
                     residual_fraction=float(sr[ix].sum()/sr.sum())))
print(json.dumps(rows),flush=True)
np.savez_compressed(ROOT/'spectral_support.npz',field_power=sf,residual_power=sr,order=order,x=x,r=r)
(ROOT/'spectral_support.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
