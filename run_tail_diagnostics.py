"""Current Hookstep endpoint: window sensitivity and matched-subspace soft modes."""
import json,time
import numpy as np
from config import ROOT,PROJECT,config
from steady_window import restore,embed,center,EmbeddedResidual
from steady_preconditioner import build_coarse,spectral_coordinates
from adaptive_solver import health

state=np.load(PROJECT/'results/steady_hookstep_20260919/hookstep.npz')
original,x=restore(config('CNT_OC',.2),state,gpu=True)
wide,xw=restore(config('CNT_OC',.2),state,wide=True,gpu=True)
n=original.n;r=original(x);rw=wide(xw);rp=embed(r,n)
norm=np.linalg.norm(r);a=state['a']
window=dict(original_residual=float(norm),wide_residual=float(np.linalg.norm(rw)),
    relative_norm_change=float(abs(np.linalg.norm(rw)/norm-1)),
    center_vector_relative_change=float(np.linalg.norm(center(rw,n)-r)/norm),
    full_vector_relative_change=float(np.linalg.norm(rw-rp)/norm),
    outer_residual_norm=float(np.linalg.norm(rw-embed(center(rw,n),n))),
    original_relative_field=float(np.linalg.norm(r[:4*n])*original.scale/np.linalg.norm(a)),
    wide_relative_field=float(np.linalg.norm(rw[:8*n])*original.scale/np.linalg.norm(a)),
    original_blocks=dict(field=float(np.linalg.norm(r[:4*n])),population=float(np.linalg.norm(r[4*n:-2])),gauge=float(np.linalg.norm(r[-2:]))),
    original_time_edge=health(a[None],.125)[0],dt_ps=.125,original_window_ps=1024,wide_window_ps=2048)
(ROOT/'endpoint_window.json').write_text(json.dumps(window,indent=2),encoding='utf8')
np.savez_compressed(ROOT/'endpoint_residuals.npz',r=r,rw=rw,a=a)
print('WINDOW',json.dumps(window),flush=True)
restrict,lift,size=spectral_coordinates(n,original.cells,384)
indices=np.r_[np.arange(385),np.arange(n-384,n)]
freq=np.fft.fftfreq(n,.125);t=(np.arange(n)-n/2)*.125
results=[]
for name,residual in [('original',original),('wide_matched_subspace',EmbeddedResidual(wide,n))]:
    rr=residual(x);start=time.perf_counter()
    def audit(matrix,rc,s,vh):
        modes=[]
        for v,sv in zip(vh[-4:][::-1],s[-4:][::-1]):
            direction=lift(v);z=(direction[:2*n]+1j*direction[2*n:4*n]).reshape(2,n)
            spectral=np.fft.fft(z,norm='ortho');power=np.sum(abs(z)**2,axis=0)
            eps=1e-6
            jn=(original(x+eps*direction)-original(x-eps*direction))/(2*eps)
            jw=(wide(xw+eps*embed(direction,n))-wide(xw-eps*embed(direction,n)))/(2*eps)
            modes.append(dict(singular_value=float(sv),field_fraction=float(power.sum()),
                field_core_fraction=float(np.sum(abs(spectral[:,abs(freq)<=.0625])**2)),
                field_wing_fraction=float(np.sum(abs(spectral[:,abs(freq)>.0625])**2)),
                time_outside_100ps_fraction=float(power[abs(t)>100].sum()),
                time_edge_fraction=float(power[abs(t)>.45*n*.125].sum()),
                population_fraction=float(np.sum(v[4*len(indices):-2]**2)),
                phase_fraction=float(v[-2]**2),time_fraction=float(v[-1]**2),
                original_full_Jv_norm=float(np.linalg.norm(jn)),wide_full_Jv_norm=float(np.linalg.norm(jw)),
                wide_outer_Jv_norm=float(np.linalg.norm(jw-embed(center(jw,n),n))),
                derivative_window_difference=float(np.linalg.norm(jw-embed(jn,n)))))
        g=matrix.T@rc
        result=dict(name=name,dimension=size,seconds=time.perf_counter()-start,
            sigma_min=float(s[-1]),sigma_max=float(s[0]),condition=float(s[0]/s[-1]),
            normalized_projected_gradient=float(np.linalg.norm(g)/(s[0]*np.linalg.norm(rc))),modes=modes)
        results.append(result)
        np.savez_compressed(ROOT/(name+'_soft_modes.npz'),vectors=vh[-4:],singular_values=s[-4:],indices=indices)
        (ROOT/'endpoint_soft_modes.json').write_text(json.dumps(results,indent=2),encoding='utf8')
        print('SOFT',json.dumps(result),flush=True)
    build_coarse(residual,x,rr,cutoff=384,audit=audit)
