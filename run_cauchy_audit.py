"""Four same-state Cauchy/Hookstep controls; no nonlinear continuation."""
import json,hashlib,time
import numpy as np
from config import ROOT,PROJECT
from steady_parameters import parameter_residual
from steady_support_lu import build_factored
from steady_hookstep import arnoldi,hookstep
from steady_cauchy import span_audit,cauchy_step,augmented_step

prior=PROJECT/'results/steady_merit_20260919'
template=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
sources=['run_cauchy_audit.py','steady_cauchy.py','steady_hookstep.py','steady_support_lu.py','steady_state.py','steady_parameters.py']
hashes={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in sources}

def derivative(f,x,d,h=1e-5):
    eps=h/max(np.linalg.norm(d),1e-100)
    return (f(x+eps*d)-f(x-eps*d))/(2*eps)

def evaluate(f,x,r,step,pred):
    js=derivative(f,x,step,1e-6);independent=float(-np.dot(r,js)-.5*np.dot(js,js))
    feasible=bool(f.feasible(x+step));rr=f(x+step) if feasible else None
    actual=float(.5*(np.dot(r,r)-np.dot(rr,rr))) if feasible else None
    return dict(model_prediction=float(pred),independent_prediction=independent,feasible=feasible,
        actual_reduction=actual,rho=actual/pred if feasible and pred>0 else None,
        residual=float(np.linalg.norm(rr)) if feasible else None,
        residual_decrease=1-float(np.linalg.norm(rr)/np.linalg.norm(r)) if feasible else None,
        step_norm=float(np.linalg.norm(step)))

for name in ['down_270','down_275']:
    old=json.loads((prior/(name+'.json')).read_text());input_path=PROJECT/old['input']
    assert hashlib.sha256(input_path.read_bytes()).hexdigest()==old['input_sha256']
    x=np.load(input_path)['x'];d=np.load(prior/(name+'_directions.npz'))['gradient_direction']
    f=parameter_residual(template,'pump',(old['pump_W']-.05)/.005,gpu=True);r=f(x);weights=np.ones_like(x)
    assert abs(np.linalg.norm(r)-old['physical_residual'])<1e-14
    print('BUILD',name,flush=True);start=time.perf_counter();pre,info=build_factored(f,x,r,'C')
    h,z,y,linear,v=arnoldi(lambda s:derivative(f,x,s),r,pre,limit=240,tolerance=.008,return_basis=True)
    span,coeff,singular=span_audit(z,d,weights);jd=derivative(f,x,d)
    beta=np.linalg.norm(r);target=np.r_[beta,np.zeros(h.shape[0]-1)]
    local=ROOT/(name+'_basis.npz');np.savez_compressed(local,h=h,z=z,v=v)
    record=dict(name=name,pump_W=old['pump_W'],input=old['input'],input_sha256=old['input_sha256'],
        gradient_input=str((prior/(name+'_directions.npz')).relative_to(PROJECT)).replace('\\','/'),
        gradient_input_sha256=hashlib.sha256((prior/(name+'_directions.npz')).read_bytes()).hexdigest(),
        physical_residual=float(beta),krylov=len(y),linear_prediction=linear,
        linear_check=float(np.linalg.norm(r+derivative(f,x,z@y,1e-6))/beta),coarse=info,span=span,
        smallest_relative_state_singular=float(singular[-1]/singular[0]),
        output_basis_orthogonality=float(np.linalg.norm(v.T@v-np.eye(v.shape[1]))),
        basis_local_sha256=hashlib.sha256(local.read_bytes()).hexdigest(),
        code_sha256=hashes,trials=[])
    saved=dict(direction_cauchy=d,state_singular_values=singular,projection_coefficients=coeff['raw'],
        projected_direction=z@coeff['raw'],h=h)
    for i,radius in enumerate([.00156,.003125,.00625]):
        sc,pc,alpha=cauchy_step(r,d,jd,radius,weights)
        yc=alpha*coeff['raw'];represented=z@yc
        represented_j=derivative(f,x,represented,1e-6)
        reduced_norm=float(np.linalg.norm(target-h@yc));full_norm=float(np.linalg.norm(r+represented_j))
        representation=dict(step_relative_error=float(np.linalg.norm(sc-represented)/np.linalg.norm(sc)),
            represented_step_norm=float(np.linalg.norm(represented)),reduced_norm=reduced_norm,
            independent_norm=full_norm,relative_model_norm_difference=abs(full_norm-reduced_norm)/beta,
            represented_prediction=float(.5*(beta**2-reduced_norm**2)),cauchy_prediction=pc)
        sh,ph,_=hookstep(h,z,beta,radius,weights)
        sa,guard,raw=augmented_step(h,z,v,r,d,jd,radius,weights)
        eh=evaluate(f,x,r,sh,ph);ec=evaluate(f,x,r,sc,pc);ea=evaluate(f,x,r,raw,guard['raw_prediction'])
        choice='cauchy' if ec['independent_prediction']>eh['independent_prediction'] else 'hookstep'
        row=dict(radius=radius,representation=representation,hookstep=eh,cauchy=ec,
            safeguard_choice=choice,safeguard=ec if choice=='cauchy' else eh,
            augmented=ea,guard=guard,acceptance=dict(model_cauchy=guard['raw_model_pass'],
                actual_rho=bool(ea['rho'] is not None and ea['rho']>.1),
                beats_cauchy_actual=bool(ea['residual'] is not None and ea['residual']<ec['residual'])))
        record['trials'].append(row);saved.update({f'hookstep_{i}':sh,f'cauchy_{i}':sc,f'augmented_{i}':raw})
    record.update(seconds=time.perf_counter()-start,evaluations=f.evaluations)
    np.savez_compressed(ROOT/(name+'_directions.npz'),**saved)
    (ROOT/(name+'.json')).write_text(json.dumps(record,indent=2))
    print('RESULT',name,json.dumps(dict(span=span,minimum_singular_ratio=record['smallest_relative_state_singular'],trials=record['trials'])),flush=True)
