import unittest
import numpy as np
from steady_gkb_globalized import radius_search,solve_outer

class QuadraticResidual:
    def __call__(self,x):return np.array([1-x[0]+8*x[0]**2])
    def feasible(self,x):return bool(np.all(np.isfinite(x)))

class Rosenbrock:
    def __call__(self,x):return np.array([10*(x[1]-x[0]**2),1-x[0]])
    def feasible(self,x):return bool(np.all(np.isfinite(x)))

class TestGlobalized(unittest.TestCase):
    def test_true_merit_radius_selection(self):
        f=QuadraticResidual();x=np.zeros(1);r=f(x)
        derivative=lambda v,h:v*-1
        for radius in [.125,.02]:
            chosen,trials,t=radius_search(f,x,r,np.eye(1),-np.eye(1),-np.eye(1),-r,radius,derivative,np.ones(1),-np.ones(1))
            self.assertIsNotNone(chosen)
            best=trials[chosen[0]]
            self.assertAlmostEqual(best['actual'],max(v['actual'] for v in trials if v['eligible']))
            self.assertEqual(len(best['checks']),3)
            self.assertGreater(len(trials),1)
        self.assertTrue(any(v['radius']>.02 for v in trials))
    def test_both_arms_known_root(self):
        f=Rosenbrock()
        jt=lambda x,v:np.array([-20*x[0]*v[0]-v[1],10*v[0]])
        for arm in ['fresh','recycled']:
            x,rows,status=solve_outer(f,jt,np.array([-1.2,1.]),arm,old_basis=np.eye(2),maximum=2,max_steps=80,radius=.5)
            self.assertEqual(status,'residual_converged')
            np.testing.assert_allclose(x,[1,1],atol=1e-6)
            for row in rows:
                chosen=row['trials'][row['accepted_trial']]
                self.assertTrue(chosen['passed'])
                self.assertGreaterEqual(chosen['prediction'],chosen['cauchy_prediction']*(1-1e-6))
if __name__=='__main__':unittest.main()
