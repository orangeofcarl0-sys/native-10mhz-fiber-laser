"""Single endpoint, four enrichment controls, frozen comparison protocol."""
import json,time,hashlib
import numpy as np
from config import ROOT,PROJECT
from steady_descent_sources import physical_residual,annulus_coordinates,streamed_gradient,enriched_model,guarded_step,BASE,SCALES
from steady_support_lu import build_factored
from steady_hookstep import arnoldi
from steady_cauchy import cauchy_step,augmented_step

start=time.perf_counter();radii=[.00156,.003125,.00625];scales=[1e-5,3e-6,1e-6]
template=np.load(PROJECT/'results/steady_tail_20260919/tail_continuation.npz')
source=PROJECT/'results/steady_step_gate_20260920/continuation_final.npz';x=np.load(source)['x']
f=physical_residual(template,gpu=True);r=f(x);assert abs(np.linalg.norm(r)-.0012081439204973431)<1e-14
result=dict(input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),initial_residual=float(np.linalg.norm(r)),radii=radii,parameter_base=BASE,parameter_scales=SCALES,continuation_threshold=1.25,parameters={},directions={},trials=[])
def save():
    (ROOT/'source_audit.json').write_text(json.dumps(result,indent=2))
def derivative(d,h=1e-5):
    eps=h/max(np.linalg.norm(d),1e-100)
    return (f(x+eps*d)-f(x-eps*d))/(2*eps)
def direction_info(d):
    js=[derivative(d,h) for h in scales];j=js[0];slope=float(r@j);curvature=float(j@j)
    return j,dict(slope=slope,response_norm=float(np.linalg.norm(j)),cosine=float(-slope/(np.linalg.norm(r)*np.linalg.norm(j))),check_relative=[float(np.linalg.norm(v-j)/max(np.linalg.norm(j),1e-100)) for v in js],free_alpha=-slope/curvature,free_prediction=slope*slope/(2*curvature),free_relative_merit_prediction=slope*slope/(curvature*float(r@r)),descent=slope<0)
print('CURRENT CORE BUILD',flush=True);cache={};core_audit={}
pre,build=build_factored(f,x,r,'C',descent=cache,audit=core_audit)
dc=-cache['lift'](cache['gradient']);dc/=np.linalg.norm(dc);jc,info=direction_info(dc);result['directions']['core']=info
result['core_gradient_norm']=float(np.linalg.norm(cache['gradient']))
result['core_central_gradient_norm']=float(np.linalg.norm(core_audit['gradient']))
gc_full=cache['lift'](core_audit['gradient'])
result['core_gradient_blocks']=dict(field=float(np.linalg.norm(gc_full[:4*f.n])),population=float(np.linalg.norm(gc_full[4*f.n:-2])),gauge=float(np.linalg.norm(gc_full[-2:])))
result['core_gradient_stencil_difference']=float(np.linalg.norm(core_audit['gradient']-cache['gradient'])/np.linalg.norm(core_audit['gradient']))
print('ARNOLDI',flush=True);H,Z,y,linear,V=arnoldi(derivative,r,pre,limit=240,tolerance=.008,return_basis=True)
result['common_basis']=dict(dimension=len(y),internal_error=linear,build=build)
print('ANNULUS STREAM',flush=True);restrict,lift,size=annulus_coordinates(f.n,f.cells)
ga,gf=streamed_gradient(f,x,r,lift,size);da=-lift(ga);da/=np.linalg.norm(da);ja,info=direction_info(da)
result['directions']['annulus']=info;result['annulus']=dict(dimension=size,gradient_norm=float(np.linalg.norm(ga)),forward_central_difference=float(np.linalg.norm(ga-gf)/np.linalg.norm(ga)),inner_GHz=375,outer_GHz=750)
print('ANNULUS',info,flush=True)
recovery=json.loads((ROOT/'history_recovery.json').read_text());assert recovery['all_30_residuals_match']
dh=np.load(ROOT/'history_directions.npz')['directions'][-3:];jh=[]
for i,d in enumerate(dh):
    j,info=direction_info(d);jh.append(j);result['directions']['history_'+str(i)]=info
responses={}
for name in BASE:
    columns=[]
    for h in [1e-3,5e-4]:
        plus=physical_residual(template,{name:h},gpu=True);minus=physical_residual(template,{name:-h},gpu=True)
        assert [v[0] for v in plus.engine.ops]==[v[0] for v in minus.engine.ops]
        columns.append((plus(x)-minus(x))/(2*h))
    j=columns[0];responses[name]=j;slope=float(r@j);curvature=float(j@j);dq=-slope/curvature
    bounded=float(np.clip(dq,-.003125,.003125));pred=float(-bounded*slope-.5*bounded*bounded*curvature)
    result['parameters'][name]=dict(slope=slope,response_norm=float(np.linalg.norm(j)),free_scaled_step=dq,free_physical_step=dq*SCALES[name],free_prediction=slope*slope/(2*curvature),bounded_prediction=pred,response_blocks=dict(field=float(np.linalg.norm(j[:4*f.n])),population=float(np.linalg.norm(j[4*f.n:-2])),gauge=float(np.linalg.norm(j[-2:]))),column_crosscheck=float(np.linalg.norm(columns[1]-j)/np.linalg.norm(j)))
    print('PARAMETER',name,result['parameters'][name],flush=True)
valid_parameters=[name for name in BASE if result['parameters'][name]['column_crosscheck']<.01]
assert valid_parameters,'No consistent physical parameter column'
best=max(valid_parameters,key=lambda name:result['parameters'][name]['bounded_prediction']);result['selected_parameter']=best
result['history_direction_gram']=np.column_stack([dc,dh.T]).T.dot(np.column_stack([dc,dh.T])).tolist()
models={
'S1':enriched_model(H,Z,V,r,dc[:,None],jc[:,None]),
'S2':enriched_model(H,Z,V,r,np.column_stack([dc,da]),np.column_stack([jc,ja])),
'S3':enriched_model(H,Z,V,r,np.column_stack([dc,dh.T]),np.column_stack([jc,np.array(jh).T]))}
padded=np.vstack([Z,np.zeros(Z.shape[1])]);dc_pad=np.r_[dc,0.];unit=np.zeros(len(x)+1);unit[-1]=1
models['S4']=enriched_model(H,padded,V,r,np.column_stack([dc_pad,unit]),np.column_stack([jc,responses[best]]))
def evaluate(group,s):
    if group=='S4':
        probe=physical_residual(template,{best:float(s[-1])},gpu=True)
        return probe(x+s[:-1])
    return f(x+s)
steps={}
for radius in radii:
    _,pc,alpha=cauchy_step(r,dc,jc,radius,np.ones_like(x))
    for group,model in models.items():
        s,guard=guarded_step(model,r,radius,pc)
        if group=='S1':
            _,legacy,original=augmented_step(H,Z,V,r,dc,jc,radius,np.ones_like(x))
            result.setdefault('S1_assembly_check',[]).append(dict(radius=radius,relative_step_difference=float(np.linalg.norm(s-original)/np.linalg.norm(original)),relative_prediction_difference=abs(guard['prediction']-legacy['raw_prediction'])/abs(legacy['raw_prediction'])))
            s=original;guard=dict(prediction=legacy['raw_prediction'],cauchy_prediction=pc,raw_model_pass=legacy['raw_model_pass'],lambda_value=legacy['lambda_value'])
        rr=evaluate(group,s);actual=float((r@r-rr@rr)/2);pred=guard['prediction'];checks=[]
        for h in scales:
            eps=h/max(np.linalg.norm(s),1e-100);js=(evaluate(group,eps*s)-evaluate(group,-eps*s))/(2*eps)
            checked=float(-r@js-.5*(js@js));checks.append(dict(h=h,prediction=checked,discrepancy=abs(checked-pred)/max(abs(pred),1e-100)))
        passed=bool(guard['raw_model_pass'] and f.feasible(x+(s[:-1] if group=='S4' else s)) and actual>0 and actual/pred>.1 and all(v['prediction']>0 and v['discrepancy']<.05 for v in checks))
        row=dict(group=group,radius=radius,**guard,actual_reduction=actual,rho=actual/pred if pred>0 else None,residual=float(np.linalg.norm(rr)),step_norm=float(np.linalg.norm(s)),checks=checks,passed=passed,parameter_step=float(s[-1]) if group=='S4' else 0.,physical_parameter_value=BASE[best]+SCALES[best]*float(s[-1]) if group=='S4' else None)
        result['trials'].append(row);steps[group+'_'+str(radius)]=s
        print('TRIAL',group,radius,'actual',actual,'rho',row['rho'],'pass',passed,flush=True)
        save()
qualified=[]
for row in result['trials']:
    base=next(v for v in result['trials'] if v['group']=='S1' and v['radius']==row['radius'])
    row['actual_gain_over_S1']=row['actual_reduction']/base['actual_reduction'] if base['actual_reduction']>0 else None
    if row['group']!='S1' and row['passed'] and base['passed'] and row['actual_gain_over_S1']>=1.25:qualified.append(row)
result['qualified']=[dict(group=v['group'],radius=v['radius'],gain=v['actual_gain_over_S1'],actual_reduction=v['actual_reduction']) for v in qualified]
result['seconds']=time.perf_counter()-start
np.savez_compressed(ROOT/'audit_vectors.npz',core=dc,annulus=da,history=dh,**steps)
save();print('QUALIFIED',result['qualified'],flush=True)
