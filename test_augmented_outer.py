import unittest
from unittest.mock import patch
import numpy as np
from steady_augmented_outer import solve_augmented,next_radius
from steady_support_lu import build_factored
from steady_support import SUPPORTS

class OuterTests(unittest.TestCase):
    def test_streamed_gradient_needs_no_extra_maps(self):
        class F:
            n=8;cells=1;dt=.5
            calls=0
            def __call__(self,x):return 2*x+.1
            def batch(self,x):self.calls+=len(x);return 2*x+.1
        f=F();x=np.linspace(.1,.5,35);r=f(x);cache={}
        with patch.dict(SUPPORTS,{'tiny':(False,1)}):
            build_factored(f,x,r,'tiny',descent=cache)
        self.assertEqual(f.calls,len(cache['gradient']))
        np.testing.assert_allclose(cache['gradient'],2*cache['restrict'](r),atol=1e-8)

    def test_paired_linear_root_and_fresh_cheap_direction(self):
        class F:
            def __call__(self,x):return np.arange(1,7)*x
            def feasible(self,x):return True
        f=F();a=np.diag(np.arange(1.,7.));calls=[]
        def builder(f,x,r,cache):
            calls.append(x.copy());cache.update(matrix=a,restrict=lambda r:r,lift=lambda g:g,gradient=a.T@r)
            return np.linalg.inv(a),{}
        for arm in (False,True):
            x,history,status=solve_augmented(f,np.ones(6)*.01,arm,max_steps=15,builder=builder,linear_limit=6)
            self.assertEqual(status,'residual_converged');self.assertLess(np.linalg.norm(f(x)),1e-7)
            if arm:
                self.assertTrue(any(h['gradient_source']=='cheap_current_residual' for h in history))
                for h in history:
                    self.assertLess(h['checked_descent_slope'],0)
                    t=h['trials'][h['accepted_trial']];self.assertGreaterEqual(t['G_C'],1-1e-6)

    def test_non_descent_cheap_direction_forces_current_rebuild(self):
        class F:
            def __call__(self,x):return 2*x
            def feasible(self,x):return True
        def builder(f,x,r,cache):
            # Deliberately stale/wrong coarse gradient, but valid fresh full gradient.
            cache.update(matrix=-2*np.eye(3),restrict=lambda r:r,lift=lambda g:g,gradient=2*r)
            return .5*np.eye(3),{}
        x,h,status=solve_augmented(F(),np.ones(3)*.01,max_steps=10,builder=builder,linear_limit=3)
        self.assertEqual(status,'residual_converged')
        self.assertTrue(any('non_descent' in b['reasons'] for row in h for b in row['builds']))

    def test_nonlinear_rosenbrock_with_rejected_trials(self):
        class F:
            def __call__(self,x):return np.array([10*(x[1]-x[0]**2),1-x[0]])
            def feasible(self,x):return True
        def builder(f,x,r,cache):
            a=np.array([[-20*x[0],10.],[-1.,0.]])
            cache.update(matrix=a,restrict=lambda r:r,lift=lambda g:g,gradient=a.T@r)
            return np.linalg.inv(a),{}
        x,h,status=solve_augmented(F(),np.array([-1.2,1.]),max_steps=80,radius=.1,builder=builder,linear_limit=2)
        self.assertEqual(status,'residual_converged')
        np.testing.assert_allclose(x,[1,1],atol=1e-7)
        self.assertTrue(any(len(row['trials'])>1 for row in h))

    def test_trust_radius_can_exceed_old_single_step_range(self):
        self.assertEqual(next_radius(.00625,.96,.00625),.0125)
        self.assertEqual(next_radius(.0125,.1,.0125),.003125)

if __name__=='__main__':unittest.main()
