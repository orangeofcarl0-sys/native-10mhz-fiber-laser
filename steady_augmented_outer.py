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
                    builder=None,observer=None,linear_limit=240,gate_policy="newton"):
    """Same physical equation and trust controller for both arms; D=I.

    With gate_policy="step", constrained candidates require agreement within 5%
    at two independent difference scales; the Newton gate applies only inside
    the current radius. The legacy default preserves previous experiments.
    The 1% gate checks the unconstrained Newton direction using independent Jv.
    A fresh builder is retried before stopping on this gate. No minimum-progress
    stop is used. Model discrepancy and low rho trigger next-state rebuilding.
    """
    if gate_policy not in {"newton","step"}:raise ValueError("Unknown gate policy")
    x=initial.copy();r=residual(x);history=[];refresh=['initial'];cache={}
    if builder is None:
        builder=lambda f,x,r,cache:build_factored(f,x,r,'C',descent=cache)
    def save():
        if observer:observer(x,r,history)
    status='iteration_budget_reached'
    for iteration in range(max_steps):
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
                g=cache['gradient'] if rebuilt else cache['matrix'].T@cache['restrict'](r)
                d=-cache['lift'](g);d/=max(np.linalg.norm(d),1e-100)
                jd=derivative(d);jd_check=derivative(d,1e-6)
                slope=float(np.dot(r,jd));slope_check=float(np.dot(r,jd_check))
                row.update(gradient_source='full_output_rebuild' if rebuilt else 'cheap_current_residual',
                    descent_slope=slope,checked_descent_slope=slope_check)
                if max(slope,slope_check)>=0:
                    if not rebuilt:refresh=['non_descent'];continue
                    status='no_verified_descent';break
                spans,_,_=span_audit(z,d,np.ones_like(x))
                row.update(span=spans,jd_check_relative=float(np.linalg.norm(jd-jd_check)/max(np.linalg.norm(jd_check),1e-100)))
            break
        row.update(precondition_rebuilt=rebuilt,newton_norm=float(np.linalg.norm(newton)),linear_gate=eta)
        if status!='iteration_budget_reached':save();break
        accepted=False;weights=np.ones_like(x)
        for attempt in range(14):
            used_radius=radius
            sh,ph,_=hookstep(h,z,norm,radius,weights)
            if augmented:
                sc,pc,alpha=cauchy_step(r,d,jd,radius,weights)
                step,guard,raw=augmented_step(h,z,v,r,d,jd,radius,weights)
                pred=guard['selected_prediction'];kind='cauchy_model_fallback' if guard['fallback'] else 'augmented'
            else:step,pred,kind=sh,ph,'original'
            rr,t=trial(step,pred);t.update(attempt=attempt,radius=float(used_radius),kind=kind,hookstep_prediction=ph)
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
