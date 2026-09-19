"""Positive output energy closes the free-pump stationary cavity equations."""
import numpy as np
from steady_bordered import BorderedResidual

class EnergyResidual(BorderedResidual):
    def __init__(self,state,target_nJ=.9,gpu=False,q_bounds=(-9,2)):
        if target_nJ<=0:raise ValueError('Energy target must be positive')
        normal=np.zeros_like(state['x']);normal[0]=1.
        super().__init__(state,'pump',normal,gpu,q_bounds)
        self.target_nJ=target_nJ

    def __call__(self,y):
        r=super().__call__(y)
        self.energy_nJ=float(np.sum(abs(self.output)**2)*self.dt/1000)
        r[-1]=(self.energy_nJ-self.target_nJ)/self.target_nJ
        return r

    def batch(self,states):
        answer=np.empty_like(states)
        for q in np.unique(states[:,-1]):
            mask=states[:,-1]==q;engine=self.get(q)
            answer[mask,:-1]=engine.batch(states[mask,:-1])
            answer[mask,-1]=(engine.batch_output_energy_nJ-self.target_nJ)/self.target_nJ
        self.evaluations+=len(states)
        return answer
