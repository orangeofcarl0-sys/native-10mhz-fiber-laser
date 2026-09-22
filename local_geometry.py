"""Local objective-curvature probes in a fixed real state metric."""
import numpy as np


def curvature_probe(value_gradient,x,r,d,jv,jtv,scales=(1e-4,3e-5,1e-5,3e-6,1e-6)):
    d=np.asarray(d)/np.linalg.norm(d);jd=jv(d);gn=jtv(jd);qgn=float(jd@jd)
    if qgn<=0:raise ValueError('Zero GN curvature; ratio undefined')
    rows=[];hessians=[]
    for h in scales:
        rp,gp=value_gradient(x+h*d);rm,gm=value_gradient(x-h*d)
        hv=(gp-gm)/(2*h);qfull=float(d@hv);missing=qfull-qgn
        rows.append(dict(h=h,gn=qgn,full=qfull,missing=missing,kappa=abs(missing)/qgn,
          missing_vector_ratio=float(np.linalg.norm(hv-gn)/max(np.linalg.norm(gn),1e-100)),
          scalar_objective_curvature=float((rp@rp-2*r@r+rm@rm)/(2*h*h)),
          residual_weighted_curvature=float(r@(rp-2*r+rm)/(h*h))))
        hessians.append(hv)
    return rows,dict(direction=d,jv=jd,gn=gn,hv=np.array(hessians)),float(abs(d@gn-qgn)/qgn)
