import unittest
from unittest.mock import patch
import numpy as np
from steady_augmented_outer import solve_augmented,next_radius
from steady_support_lu import build_factored
from steady_support import SUPPORTS

class OuterTests(unittest.TestCase):
    def test_full_gradient_provider_with_preconditioner_only_cache(self):
        class F:
            def __call__(self,x):return np.arange(1.,7.)*x
            def feasible(self,x):return True
        f=F();a=np.arange(1.,7.);calls=[]
        def provider(x):calls.append(x.copy());return f(x),a*f(x)
        def builder(f,x,r,cache):return np.diag(1/a),{}
        x,h,status=solve_augmented(f,np.ones(6)*.01,max_steps=12,radius=.001,
              builder=builder,linear_limit=6,gate_policy='step',value_gradient=provider)
        self.assertEqual(status,'residual_converged')
        self.assertTrue(any(not row['precondition_rebuilt'] for row in h))
        for row in h:
            self.assertEqual(row['gradient_source'],'full_adjoint')
            self.assertAlmostEqual(row['gradient_norm'],np.linalg.norm(a*f(calls[row['step']])))
            t=row['trials'][row['accepted_trial']]
            self.assertTrue(t['candidate_model_pass']);self.assertGreaterEqual(t['G_C'],1-1e-6)

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
        x,h,status=solve_augmented(F(),np.array([-1.2,1.]),max_steps=80,radius=.1,builder=builder,linear_limit=2,gate_policy='step')
        self.assertEqual(status,'residual_converged')
        np.testing.assert_allclose(x,[1,1],atol=1e-7)
        self.assertTrue(all(row['trials'][row['accepted_trial']]['candidate_model_pass'] for row in h))


    def test_enrichment_uses_current_responses_and_reaches_known_root(self):
        class F:
            def __call__(self,x):return np.array([10*(x[1]-x[0]**2),1-x[0]])
            def feasible(self,x):return True
        def builder(f,x,r,cache):
            a=np.array([[-20*x[0],10.],[-1.,0.]])
            cache.update(matrix=a,restrict=lambda r:r,lift=lambda g:g,gradient=a.T@r)
            return np.linalg.inv(a),{}
        x,h,status=solve_augmented(F(),np.array([-1.2,1.]),max_steps=80,radius=.1,builder=builder,linear_limit=2,gate_policy='step',enrichment=lambda f,x,r,d,rebuilt:np.eye(2))
        self.assertEqual(status,'residual_converged')
        np.testing.assert_allclose(x,[1,1],atol=1e-7)
        self.assertTrue(all(row['extra_direction_count']==2 for row in h))
        self.assertTrue(all(row['trials'][row['accepted_trial']]['candidate_model_pass'] for row in h))
        self.assertFalse(np.allclose(h[0]['extra_response_slopes'],h[-1]['extra_response_slopes']))

    def test_step_gate_ignores_unexecuted_bad_newton(self):
        class F:
            def __call__(self,x):return x
            def feasible(self,x):return True
        def builder(f,x,r,cache):
            cache.update(matrix=np.eye(2),restrict=lambda r:r,lift=lambda g:g,gradient=r)
            return np.eye(2),{}
        from steady_hookstep import arnoldi as real_arnoldi
        def bad_newton(*args,**kwargs):
            h,z,y,eta,v=real_arnoldi(*args,**kwargs)
            return h,z,y*100,eta,v
        with patch('steady_augmented_outer.arnoldi',side_effect=bad_newton):
            _,old,status=solve_augmented(F(),np.ones(2),max_steps=1,builder=builder)
            self.assertEqual(status,'linear_accuracy_limited')
            x,new,status=solve_augmented(F(),np.ones(2),max_steps=1,builder=builder,gate_policy='step')
            self.assertIn('accepted_trial',new[0])
            self.assertLess(np.linalg.norm(x),np.sqrt(2))
            self.assertTrue(new[0]['trials'][new[0]['accepted_trial']]['candidate_model_pass'])
            _,near,status=solve_augmented(F(),np.ones(2),max_steps=1,radius=200,builder=builder,gate_policy='step')
            self.assertEqual(status,'linear_accuracy_limited')

    def test_step_gate_rejects_inconsistent_candidate_model(self):
        class F:
            def __call__(self,x):return x
            def feasible(self,x):return True
        def builder(f,x,r,cache):
            cache.update(matrix=np.eye(2),restrict=lambda r:r,lift=lambda g:g,gradient=r)
            return np.eye(2),{}
        from steady_cauchy import augmented_step as original
        def biased_model(*args,**kwargs):
            step,guard,raw=original(*args,**kwargs)
            guard['selected_prediction']*=1.2
            return step,guard,raw
        with patch('steady_augmented_outer.augmented_step',side_effect=biased_model):
            x,h,status=solve_augmented(F(),np.ones(2),max_steps=1,radius=.01,builder=builder,gate_policy='step')
        trials=h[0]['trials']
        self.assertFalse(trials[0]['candidate_model_pass'])
        self.assertLess(trials[1]['radius'],trials[0]['radius'])
        if 'accepted_trial' in h[0]:
            accepted=trials[h[0]['accepted_trial']]
            self.assertTrue(accepted['candidate_model_pass'])
            self.assertEqual(accepted['kind'],'cauchy_actual_fallback')

    def test_trust_radius_can_exceed_old_single_step_range(self):
        self.assertEqual(next_radius(.00625,.96,.00625),.0125)
        self.assertEqual(next_radius(.0125,.1,.0125),.003125)

if __name__=='__main__':unittest.main()
