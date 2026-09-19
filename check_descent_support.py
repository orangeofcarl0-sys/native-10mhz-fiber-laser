"""Orthogonal support accounting for saved candidate steps; no new simulation."""
import json
import numpy as np
from config import ROOT,PROJECT
from steady_preconditioner import spectral_coordinates
from steady_descent_sources import annulus_coordinates
v=np.load(ROOT/'audit_vectors.npz');t=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
n=t['original_template'].shape[-1];cells=len(v['core'])-4*n-2
rc,lc,_=spectral_coordinates(n,cells,768);ra,la,_=annulus_coordinates(n,cells)
rows=[];overlaps=[]
for radius in [.00156,.003125,.00625]:
    for group in ['S1','S2','S3','S4']:
        raw=v[group+'_'+str(radius)];s=raw[:-1] if group=='S4' else raw
        core=lc(rc(s));ann=la(ra(s));outer=s-core-ann;fn=np.linalg.norm(s[:4*n])
        partition=float((np.linalg.norm(core[:4*n])**2+np.linalg.norm(ann)**2+np.linalg.norm(outer)**2)/fn**2)
        assert abs(partition-1)<1e-12
        rows.append(dict(group=group,radius=radius,field_step_norm=float(fn),population_step_norm=float(np.linalg.norm(s[4*n:-2])),gauge_step_norm=float(np.linalg.norm(s[-2:])),core_field_fraction=float(np.linalg.norm(core[:4*n])/fn),annulus_field_fraction=float(np.linalg.norm(ann)/fn),outer_field_fraction=float(np.linalg.norm(outer)/fn),orthogonal_partition=partition))
    s2=v['S2_'+str(radius)];s3=v['S3_'+str(radius)];aa=ra(s2);bb=ra(s3)
    overlaps.append(dict(radius=radius,state_cosine=float(s2@s3/np.linalg.norm(s2)/np.linalg.norm(s3)),annulus_cosine=float(aa@bb/np.linalg.norm(aa)/np.linalg.norm(bb))))
(ROOT/'step_support.json').write_text(json.dumps(dict(partitions=rows,overlaps=overlaps),indent=2))
print('Candidate support partitions verified')
