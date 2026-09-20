import unittest
import numpy as np
from steady_gkb import gkb
from steady_gkb_outer import solve_gkb


class Residual:
    def __call__(self,x):return np.array([10*(x[1]-x[0]**2),1-x[0]])
    def feasible(self,x):return bool(np.all(np.isfinite(x)))


class TestPureGKB(unittest.TestCase):
    def test_adaptive_stops_after_two_small_increments(self):
        diag=np.linspace(1,2,80);b=np.random.default_rng(7).normal(size=80)
        *_,rows=gkb(lambda x:diag*x,lambda x:diag*x,b,64,1.)
        self.assertTrue(rows[-1]['saturated'])
        self.assertGreaterEqual(rows[-1]['k'],24)
        self.assertLess(rows[-1]['k'],64)
        checks=[r for r in rows if 'prediction' in r]
        self.assertTrue(all(0<=r['marginal_gain']<.01 for r in checks[-2:]))

    def test_nonlinear_root_and_guarded_rejections(self):
        f=Residual()
        def jt(x,v):return np.array([-20*x[0]*v[0]-v[1],10*v[0]])
        x,rows,status=solve_gkb(f,jt,np.array([-1.2,1.]),radius=1.,max_steps=40)
        self.assertEqual(status,'residual_converged')
        np.testing.assert_allclose(x,[1,1],atol=1e-6)
        self.assertTrue(any(len(r['trials'])>1 for r in rows))
        for row in rows:
            t=row['trials'][row['accepted_trial']]
            self.assertTrue(t['passed'])
            self.assertGreaterEqual(t['prediction'],t['cauchy_prediction']*(1-1e-6))
            self.assertEqual(len(t['checks']),3)


if __name__=='__main__':unittest.main()
