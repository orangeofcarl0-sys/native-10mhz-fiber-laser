import unittest
import numpy as np
from config import config,map_initial
from steady_state import CavityResidual
from steady_parameters import parameter_residual
from steady_bordered import BorderedResidual
from steady_diagnostics import blocks


class ParameterTests(unittest.TestCase):
    def state(self):
        c=config('CNT_OC',.2);a,p,_=map_initial(c,[.8],n=256,dt=.5)
        r=CavityResidual(c,.5,a[0],.05,.8)
        return dict(x=r.pack(a[0],p[0]),a=a[0],pop=p[0],original_template=a[0],
            gauge_template=r.template,time_tangent=r.time_tangent,scale=r.scale,time_scale=r.time_scale,dt=.5)

    def test_baseline_border_and_batch(self):
        state=self.state();normal=np.random.default_rng(23).normal(size=len(state['x']))
        b=BorderedResidual(state,'pump',normal)
        y=np.r_[state['x'],0.];expected=parameter_residual(state,'pump')(state['x'])
        np.testing.assert_allclose(b(y),np.r_[expected,0],atol=1e-14)
        probe=y.copy();probe[0]+=.0001;probe[-1]=.001
        np.testing.assert_allclose(b.batch(np.array([y,probe])),[b(y),b(probe)],atol=1e-13)
        d=np.zeros_like(y);d[-1]=.3;info=blocks(b,d)
        self.assertEqual(info['population_rms'],0);self.assertAlmostEqual(info['parameter_physical'],.0015)
        beyond=y.copy();beyond[-1]=-2.5
        self.assertFalse(b.feasible(beyond))
        expanded=BorderedResidual(state,'pump',normal,q_bounds=(-9,2))
        self.assertTrue(expanded.feasible(beyond))

    def test_parameter_physics_and_units(self):
        state=self.state();g=parameter_residual(state,'gdd',1).engine.c
        self.assertAlmostEqual(g['segments'][:,0].sum(),20.42)
        self.assertAlmostEqual(np.dot(g['segments'][:,0],g['segments'][:,1]),.22)
        c=parameter_residual(state,'psat',1).engine.c
        self.assertAlmostEqual(c['sa_saturation_energy_pJ']/c['sa_recovery_ps'],44)

    def test_simple_fold_border_nonsingular(self):
        # R=x²+p at the fold x=p=0; fixed-p J=0 but [J Rp; v 0] invertible.
        border=np.array([[0.,1.],[1.,0.]])
        answer=np.linalg.solve(border,[-.01,0.])
        np.testing.assert_allclose(answer,[0,-.01])


if __name__=='__main__':unittest.main()
