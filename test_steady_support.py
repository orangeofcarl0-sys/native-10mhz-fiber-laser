import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse.linalg import aslinearoperator
from steady_support import residual_support
from steady_hookstep import arnoldi,solve_globalized

class SupportTests(unittest.TestCase):
    def test_orthogonal_residual_partition(self):
        class Grid:n=4096;cells=3;dt=.5
        r=np.random.default_rng(42).normal(size=4*Grid.n+5)
        for label in ['A','B','C']:
            info=residual_support(Grid(),r,label)
            self.assertAlmostEqual(info['squared_partition'],1.,places=12)
            self.assertLess(abs(info['orthogonality']),1e-12)
        self.assertEqual(residual_support(Grid(),r,'C')['temporal_relative'],0)

    def test_prefix_residual_is_same_as_independent_arnoldi(self):
        matrix=np.diag(np.arange(1,7.))+.03*np.ones((6,6));r=np.ones(6)
        pre=aslinearoperator(np.eye(6));h,z,_,_=arnoldi(lambda v:matrix@v,r,pre,limit=5,tolerance=0)
        k=3;b=np.r_[np.linalg.norm(r),np.zeros(k)]
        y=np.linalg.lstsq(h[:k+1,:k],b,rcond=1e-12)[0]
        _,z0,y0,_=arnoldi(lambda v:matrix@v,r,pre,limit=k,tolerance=0)
        np.testing.assert_allclose(z[:,:k]@y,z0@y0,atol=1e-12)

    def test_linear_options_forwarded_and_root_unchanged(self):
        class Equation:
            evaluations=0
            def __call__(self,x):self.evaluations+=1;return np.array([2*x[0]-1,3*x[1]+2])
            def feasible(self,x):return True
        with patch('steady_hookstep.build_coarse',return_value=(aslinearoperator(np.eye(2)),{})),patch('steady_hookstep.gauge_geometry',return_value={}),patch('steady_hookstep.blocks',return_value={}),patch('steady_hookstep.arnoldi',wraps=arnoldi) as spy:
            x,h,status=solve_globalized(Equation(),np.zeros(2),max_steps=10,linear_limit=2,linear_tolerance=.001)
        self.assertEqual(status,'residual_converged');np.testing.assert_allclose(x,[.5,-2/3],atol=1e-7)
        self.assertEqual(spy.call_args.kwargs,dict(limit=2,tolerance=.001))

    def test_factored_inverse_matches_svd(self):
        from steady_support import SUPPORTS
        from steady_support_lu import build_factored
        from steady_window import build_localized
        rng=np.random.default_rng(4)
        class Equation:
            n=8;cells=1;dt=.5
            def __init__(self):self.a=rng.normal(size=(35,35))*.02-np.eye(35)
            def __call__(self,x):return self.a@x
            def batch(self,x):return x@self.a.T
        r=Equation();x=rng.normal(size=35);fx=r(x);v=rng.normal(size=35)
        with patch.dict(SUPPORTS,{'tiny':(True,1)}):actual,info=build_factored(r,x,fx,'tiny')
        expected,_=build_localized(r,x,fx,cutoff=1)
        np.testing.assert_allclose(actual@v,expected@v,rtol=1e-7,atol=1e-7)
        self.assertLess(info['factor_probe_residual'],1e-12)

    def test_singular_coarse_uses_svd_fallback(self):
        from steady_support import SUPPORTS
        from steady_support_lu import build_factored
        class Equation:
            n=8;cells=1;dt=.5
            def __call__(self,x):
                r=-x.copy();r[-1]=0;return r
            def batch(self,x):return np.array([self(v) for v in x])
        r=Equation();x=np.zeros(35)
        with patch.dict(SUPPORTS,{'tiny':(True,1)}):pre,info=build_factored(r,x,r(x)+1e-20,'tiny')
        self.assertEqual(info['factorization'],'SVD fallback')
        self.assertTrue(np.isfinite(pre@np.ones(35)).all())

    def test_inaccurate_linear_step_is_not_accepted(self):
        class Equation:
            evaluations=0
            def __call__(self,x):self.evaluations+=1;return x-np.ones(2)
            def feasible(self,x):return True
        with patch('steady_hookstep.build_coarse',return_value=(aslinearoperator(np.zeros((2,2))),{})),patch('steady_hookstep.gauge_geometry',return_value={}),patch('steady_hookstep.blocks',return_value={}):
            x,h,status=solve_globalized(Equation(),np.zeros(2),max_steps=5,linear_refresh_threshold=.01)
        self.assertEqual(status,'linear_accuracy_limited');np.testing.assert_array_equal(x,0)
        self.assertEqual(h[-1]['linear_attempts'][0]['true_relative'],1.)

    def test_stale_factor_is_refreshed_at_same_state(self):
        class Equation:
            evaluations=0
            def __call__(self,x):self.evaluations+=1;return x-np.ones(2)
            def feasible(self,x):return True
        counter=[0]
        def flaky(jv,r,pre,**kwargs):
            counter[0]+=1
            if counter[0]==2:return np.zeros((2,1)),np.zeros((2,1)),np.zeros(1),1.
            return arnoldi(jv,r,pre,**kwargs)
        with patch('steady_hookstep.build_coarse',return_value=(aslinearoperator(np.eye(2)),{})),patch('steady_hookstep.gauge_geometry',return_value={}),patch('steady_hookstep.blocks',return_value={}),patch('steady_hookstep.arnoldi',side_effect=flaky):
            x,h,status=solve_globalized(Equation(),np.zeros(2),max_steps=10,rebuild_every=30,linear_refresh_threshold=.01)
        self.assertEqual(status,'residual_converged')
        refreshed=[v for v in h if v.get('refreshed_for_accuracy')]
        self.assertEqual(len(refreshed),1)
        self.assertEqual(refreshed[0]['linear_attempts'][0]['true_relative'],1.)
        self.assertLess(refreshed[0]['linear_attempts'][1]['true_relative'],.01)

    def test_centered_jv_quadratic_known_root(self):
        class Equation:
            evaluations=0
            def __call__(self,x):self.evaluations+=1;return x*x-np.array([1.,4.])
            def feasible(self,x):return bool(np.all(x>0))
        with patch('steady_hookstep.build_coarse',return_value=(aslinearoperator(np.eye(2)),{})),patch('steady_hookstep.gauge_geometry',return_value={}),patch('steady_hookstep.blocks',return_value={}):
            x,h,status=solve_globalized(Equation(),np.array([1.3,2.3]),max_steps=12,
                jv_method='central',jv_epsilon=1e-5,jv_check_epsilon=1e-6,
                linear_limit=2,linear_tolerance=.001,linear_refresh_threshold=.01)
        self.assertEqual(status,'residual_converged')
        np.testing.assert_allclose(x,[1.,2.],atol=1e-7)

if __name__=='__main__':unittest.main()
