"""Local hyperplane-constrained parameter release; not arclength continuation."""
import numpy as np
from steady_parameters import parameter_residual,PARAMETERS
from steady_diagnostics import blocks


class BorderedResidual:
    """Y=(X,q), Rborder=(R(X,p0+scale*q), v^T(X-Xanchor))."""
    def __init__(self,state,name,normal,gpu=False,q_bounds=(-2.,2.)):
        self.state,self.name,self.gpu=state,name,gpu
        self.anchor=state['x'].copy();self.normal=normal/np.linalg.norm(normal)
        self.q_bounds=q_bounds
        self.base=parameter_residual(state,name,gpu=gpu)
        for key in ['n','dt','scale','time_scale','template','time_tangent','omega']:
            setattr(self,key,getattr(self.base,key))
        # spectral_coordinates treats all scalar tail coordinates uniformly.
        self.cells=self.base.cells+1
        self.cache={0.:self.base};self.evaluations=0
        self.initial_counts=[v[0] for v in self.base.engine.ops]

    def get(self,q):
        q=float(q)
        if q not in self.cache:
            if len(self.cache)>=3:self.cache.pop(next(iter(self.cache)))
            self.cache[q]=parameter_residual(self.state,self.name,q,gpu=self.gpu)
        return self.cache[q]

    def feasible(self,y):
        if not np.isfinite(y).all() or not self.q_bounds[0]<=y[-1]<=self.q_bounds[1] or not self.base.feasible(y[:-1]):return False
        if self.name=='gdd':
            # Keep the differentiation mesh topology fixed in this local pilot.
            return [v[0] for v in self.get(y[-1]).engine.ops]==self.initial_counts
        return True

    def unpack(self,y):return self.base.unpack(y[:-1])

    def __call__(self,y):
        engine=self.get(y[-1]);r=engine(y[:-1]);self.evaluations+=1
        self.output=engine.output;self.neq=engine.neq
        return np.r_[r,np.dot(self.normal,y[:-1]-self.anchor)]

    def batch(self,states):
        answer=np.empty_like(states)
        for q in np.unique(states[:,-1]):
            mask=states[:,-1]==q;part=states[mask,:-1]
            answer[mask,:-1]=self.get(q).batch(part)
            answer[mask,-1]=(part-self.anchor)@self.normal
        self.evaluations+=len(states)
        return answer

    def step_blocks(self,s):
        values=blocks(self.base,s[:-1]);values['parameter_scaled']=float(abs(s[-1]))
        values['parameter_physical']=float(abs(s[-1])*PARAMETERS[self.name][1])
        values['total']=float(np.linalg.norm(s));return values
