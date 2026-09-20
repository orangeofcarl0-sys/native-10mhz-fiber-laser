"""Paired outer loop with current-state descent checks and explicit Cauchy fallback."""
import numpy as np
from steady_hookstep import arnoldi,hookstep
from steady_cauchy import cauchy_step,augmented_step,span_audit
from steady_support_lu import build_factored


def next_radius(radius,rho,step_norm):
    if rho<.25:return radius*.25
    if rho>.75 and step_norm>.8*radius:return min(2.,2*radius)
    return radius


def solve_augmented(residual,initial,augmented=True,max_steps=40,radius=.003125,
                    builder=None,observer=None,linear_limit=240,gate_policy="newton",enrichment=None,model_factory=None,value_gradient=None):
    """Same physical equation and trust controller for both arms; D=I.

    With gate_policy="step", constrained candidates require agreement within 5%
    at two independent difference scales; the Newton gate applies only inside
    the current radius. The legacy default preserves previous experiments.
    Optional enrichment returns extra state directions; their responses are
    recomputed at every current state, and the same Cauchy safeguard is retained.
    Alternatively, model_factory constructs a current-state radius callback;
    its diagnostics are retained even when the actual Cauchy fallback is used.
    value_gradient(x), when supplied, returns current packed (R, J^T R) once
    per outer iteration. The builder then needs only supply a preconditioner;
    current descent and Cauchy candidates no longer read its gradient cache.
    The 1% gate checks the unconstrained Newton direction using independent Jv.
    A fresh builder is retried before stopping on this gate. No minimum-progress
    stop is used. Model discrepancy and low rho trigger next-state rebuilding.
    """
    if value_gradient is not None and not augmented:raise ValueError('Current gradient requires augmentation')
    if gate_policy not in {"newton","step"}:raise ValueError("Unknown gate policy")
    if enrichment is not None and not augmented:raise ValueError('Enrichment requires augmentation')
    if model_factory is not None and (not augmented or enrichment is not None):
        raise ValueError('Model factory requires augmentation without an enrichment callback')
    x=initial.copy();r=residual(x);history=[];refresh=['initial'];cache={}
    if builder is None:
        builder=lambda f,x,r,cache:build_factored(f,x,r,'C',descent=cache)
    def save():
        if observer:observer(x,r,history)
    status='iteration_budget_reached'
    for iteration in range(max_steps):
        if value_gradient is not None:
            r,current_gradient=value_gradient(x)
        norm=float(np.linalg.norm(r))
        if norm<1e-7:status='residual_converged';break
        row=dict(step=iteration,residual=norm,radius=float(radius),trials=[],builds=[])
        history.append(row)
        def derivative(v,h=1e-5):
            eps=h/max(np.linalg.norm(v),1e-100)
            return (residual(x+eps*v)-residual(x-eps*v))/(2*eps)
        def trial(step,pred):
            feasible=bool(residual.feasible(x+step));rr=residual(x+step) if feasible else None
            js=derivative(step,1e-6)
            independent=float(-np.dot(r,js)-.5*np.dot(js,js))
            actual=float(.5*(norm**2-np.dot(rr,rr))) if feasible else None
            rho=actual/pred if feasible and pred>0 else -1.
            true_rho=actual/independent if feasible and independent>0 else -1.
            checks=[independent]
            if gate_policy=='step':
                js_medium=derivative(step,3e-6)
                checks.append(float(-np.dot(r,js_medium)-.5*np.dot(js_medium,js_medium)))
            model_ok=all(abs(pred-q)<.05*max(abs(pred),1e-30) and q>0 for q in checks)
            return rr,dict(candidate_model_pass=bool(model_ok),checked_predictions=checks,feasible=feasible,prediction=float(pred),independent_prediction=independent,
                model_discrepancy=float(abs(pred-independent)/max(abs(pred),1e-30)),
                actual_reduction=actual,rho=float(rho),true_rho=float(true_rho),
                residual=float(np.linalg.norm(rr)) if feasible else None,
                step_norm=float(np.linalg.norm(step)))
        rebuilt=False
        while True:
            if refresh:
                cache={};pre,info=builder(residual,x,r,cache)
                row['builds'].append(dict(reasons=refresh,**info));refresh=[];rebuilt=True
            h,z,y,linear,v=arnoldi(derivative,r,pre,limit=linear_limit,tolerance=.008,return_basis=True)
            newton=z@y;eta=float(np.linalg.norm(r+derivative(newton,1e-6))/norm)
            row.setdefault('linear_attempts',[]).append(dict(dimension=len(y),model=linear,independent=eta))
            if eta>=.01 and (gate_policy=='newton' or np.linalg.norm(newton)<=radius):
                if not rebuilt:refresh=['linear_gate'];continue
                status='linear_accuracy_limited';break
            if augmented:
                if value_gradient is None:
                    g=cache['gradient'] if rebuilt else cache['matrix'].T@cache['restrict'](r)
                    d=-cache['lift'](g)
                else:
                    d=-current_gradient.copy()
                    row['gradient_norm']=float(np.linalg.norm(current_gradient))
                d/=max(np.linalg.norm(d),1e-100)
                jd=derivative(d);jd_check=derivative(d,1e-6)
                slope=float(np.dot(r,jd));slope_check=float(np.dot(r,jd_check))
                row.update(gradient_source=('full_adjoint' if value_gradient is not None else ('full_output_rebuild' if rebuilt else 'cheap_current_residual')),
                    descent_slope=slope,checked_descent_slope=slope_check)
                if max(slope,slope_check)>=0:
                    if not rebuilt:refresh=['non_descent'];continue
                    status='no_verified_descent';break
                spans,_,_=span_audit(z,d,np.ones_like(x))
                row.update(span=spans,jd_check_relative=float(np.linalg.norm(jd-jd_check)/max(np.linalg.norm(jd_check),1e-100)))
            break
        row.update(precondition_rebuilt=rebuilt,newton_norm=float(np.linalg.norm(newton)),linear_gate=eta)
        if status!='iteration_budget_reached':save();break
        enriched=None;custom_model=None
        if model_factory is not None:
            custom_model=model_factory(x,r,d,jd,h,z,v,derivative,rebuilt,radius,row)
        if enrichment is not None:
            from steady_descent_sources import enriched_model,guarded_step
            extra=np.asarray(enrichment(residual,x,r,d,rebuilt))
            if extra.ndim!=2 or extra.shape[0]!=len(x):raise ValueError('Extra directions have wrong shape')
            responses=np.column_stack([derivative(column) for column in extra.T])
            enriched=enriched_model(h,z,v,r,np.column_stack([d,extra]),np.column_stack([jd,responses]))
            row.update(extra_direction_count=extra.shape[1],extra_response_slopes=(r@responses).tolist())
        accepted=False;weights=np.ones_like(x)
        for attempt in range(14):
            used_radius=radius
            sh,ph,_=hookstep(h,z,norm,radius,weights)
            if augmented:
                sc,pc,alpha=cauchy_step(r,d,jd,radius,weights)
                if custom_model is not None:
                    raw,info=custom_model(radius)
                    passed=bool(info['prediction']>=pc*(1-1e-6) and np.linalg.norm(raw)<=radius*(1+1e-6))
                    step=raw if passed else sc
                    guard=dict(raw_model_pass=passed,raw_prediction=info['prediction'],cauchy_prediction=pc,
                        selected_prediction=info['prediction'] if passed else pc,fallback=not passed,
                        alpha_cauchy=alpha,lambda_value=info['lambda_value'])
                elif enriched is None:
                    step,guard,raw=augmented_step(h,z,v,r,d,jd,radius,weights)
                else:
                    raw,info=guarded_step(enriched,r,radius,pc)
                    passed=info['raw_model_pass'];step=raw if passed else sc
                    guard=dict(raw_model_pass=passed,raw_prediction=info['prediction'],cauchy_prediction=pc,
                        selected_prediction=info['prediction'] if passed else pc,fallback=not passed,
                        alpha_cauchy=alpha,lambda_value=info['lambda_value'])
                pred=guard['selected_prediction'];kind='cauchy_model_fallback' if guard['fallback'] else 'augmented'
            else:step,pred,kind=sh,ph,'original'
            rr,t=trial(step,pred);t.update(attempt=attempt,radius=float(used_radius),kind=kind,hookstep_prediction=ph)
            if custom_model is not None:t['seed_model']=info
            if augmented:t.update(guard=guard,alpha_cauchy=alpha,cauchy_prediction=pc,
                augmented_step_norm=float(np.linalg.norm(raw)),G_C=float(pred/pc),G_H=float(pred/ph) if ph>0 else None)
            row['trials'].append(t)
            if t['rho']<.25:refresh.append('low_rho')
            if t['model_discrepancy']>.05:refresh.append('model_discrepancy')
            radius=next_radius(radius,t['rho'],t['step_norm'])
            if gate_policy=='step' and not t['candidate_model_pass']:
                refresh.append('candidate_model_gate');radius=used_radius*.25
            if t['feasible'] and min(t['rho'],t['true_rho'])>.1 and t['actual_reduction']>0 and (gate_policy=='newton' or t['candidate_model_pass']):
                accepted=True;break
            # Actual Cauchy descent remains available even if the augmented trials fail.
            if augmented and attempt==13:
                rr,t=trial(sc,pc);t.update(attempt=14,radius=float(used_radius),kind='cauchy_actual_fallback',
                    alpha_cauchy=alpha,cauchy_prediction=pc,G_C=1.,G_H=float(pc/ph) if ph>0 else None)
                if custom_model is not None:t['seed_model']=info
                row['trials'].append(t)
                accepted=bool(t['feasible'] and min(t['rho'],t['true_rho'])>.1 and t['actual_reduction']>0 and (gate_policy=='newton' or t['candidate_model_pass']))
                if accepted:step=sc;break
            if radius==used_radius:radius*=.25
        refresh=list(dict.fromkeys(refresh))
        if accepted:
            row.update(accepted_trial=len(row['trials'])-1,selected_kind=t['kind'],next_radius=float(radius))
            x=x+step;r=rr
        else:status='globalization_stalled'
        save()
        if not accepted:break
        if radius<1e-10:status='radius_floor_reached';break
    if np.linalg.norm(r)<1e-7:status='residual_converged'
    return x,history,status
