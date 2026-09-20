"""Persisted seed diagnostics and bounded history maintenance; no fresh sweep."""
def prune_history(model, count, capacity, seed_index, radius):
    selected=list(range(count));events=[]
    while len(selected)>capacity:
        full=model.solve(selected+[seed_index],radius)[1]
        trials=[]
        for i in selected:
            prediction=model.solve([j for j in selected if j!=i]+[seed_index],radius)[1]
            trials.append(dict(index=i,prediction=float(prediction),loss=float(full-prediction)))
        minimum=min(v['loss'] for v in trials)
        # Chronological order is only a tie breaker at negligible model difference.
        remove=next(v['index'] for v in trials if v['loss']<=minimum+1e-10*max(abs(full),1e-100))
        selected.remove(remove);events.append(dict(full_prediction=float(full),removed=remove,trials=trials))
    return selected,events


def trigger_diagnostic(history):
    """Observe a hypothetical trigger; do not scan or claim a reseed is needed."""
    fresh=[r for r in history if r.get('precondition_rebuilt') and 'accepted_trial' in r]
    streak=0
    for r in reversed(fresh):
        if r['trials'][r['accepted_trial']]['seed_model']['gain']>=1.15:break
        streak+=1
    prior=[r for r in history[:-1] if 'accepted_trial' in r][-5:]
    gains=[1-r['trials'][r['accepted_trial']]['residual']/r['residual'] for r in prior]
    mean=sum(gains)/5 if len(gains)==5 else None
    current=history[-1]
    flag=bool(current.get('precondition_rebuilt') and streak>=3 and mean is not None and mean<.0002)
    return dict(low_gain_fresh_streak=streak,prior_five_mean_norm_gain=mean,hypothetical_trigger=flag,
                sweep_executed=False)
