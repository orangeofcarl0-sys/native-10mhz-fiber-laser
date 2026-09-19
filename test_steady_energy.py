import unittest
import numpy as np
import test_steady_parameters
from steady_energy import EnergyResidual
from steady_leakage import inverse

class EnergyTests(unittest.TestCase):
    def test_energy_batch_units_and_zero_exclusion(self):
        state=test_steady_parameters.ParameterTests().state();r=EnergyResidual(state,.9)
        y=np.r_[state['x'],0.];other=y.copy();other[-1]=-.5
        expected=np.array([r(y),r(other)])
        np.testing.assert_allclose(r.batch(np.array([y,other])),expected,atol=1e-12)
        value=r(y);self.assertAlmostEqual(value[-1],(np.sum(abs(r.output)**2)*r.dt/1000-.9)/.9)
        zero=y.copy();zero[:4*r.n]=0
        self.assertAlmostEqual(r(zero)[-1],-1.)

    def test_inverse_leakage_controls(self):
        # Projection misses a unit response: sigma=1e-6 is artificial here.
        data=(lambda x:x[:2],lambda x:np.r_[x,0.],np.eye(2),np.array([1.,1e-6]),np.eye(2),
              [dict(index=1,sigma=1e-6,full_norm=1.,leakage=1e6)])
        v=np.array([0.,1.,0.])
        baseline,_=inverse(data,3);cut,_=inverse(data,3,'truncate');cap,_=inverse(data,3,'cap')
        np.testing.assert_allclose(baseline@v,[0,1e6,0]);np.testing.assert_allclose(cut@v,0)
        np.testing.assert_allclose(cap@v,[0,1,0]);np.testing.assert_allclose(cap@np.array([0,0,1]),[0,0,-1])

    def test_baseline_matches_original_localized_inverse(self):
        from steady_leakage import geometry
        from steady_window import build_localized
        rng=np.random.default_rng(4)
        class LinearResidual:
            n=8;cells=1;dt=.5
            def __init__(self):self.a=rng.normal(size=(35,35))*.02-np.eye(35)
            def __call__(self,x):return self.a@x
            def batch(self,x):return x@self.a.T
        r=LinearResidual();x=rng.normal(size=35);fx=r(x)
        expected,_=build_localized(r,x,fx,cutoff=1)
        actual,_=inverse(geometry(r,x,fx,cutoff=1),35)
        v=rng.normal(size=35)
        np.testing.assert_allclose(actual@v,expected@v,rtol=1e-7,atol=1e-7)

if __name__=='__main__':unittest.main()
