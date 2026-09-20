import unittest
import numpy as np
from steady_active_history import HistoryModel
from steady_seed_bank import prune_history,trigger_diagnostic
from steady_augmented_outer import solve_augmented

class SeedBankTests(unittest.TestCase):
    def test_actual_cauchy_fallback_keeps_model_diagnostics(self):
        class F:
            def __call__(self,x):return x
            def feasible(self,x):return True
        def builder(f,x,r,cache):
            cache.update(matrix=np.eye(2),restrict=lambda r:r,lift=lambda g:g,gradient=r)
            return np.eye(2),{}
        def factory(x,r,d,jd,h,z,v,derivative,rebuilt,radius,row):
            return lambda rad:(np.ones(2)*rad/np.sqrt(2),dict(prediction=1.,gain=2.,lambda_value=0.))
        x,h,status=solve_augmented(F(),np.ones(2),max_steps=1,radius=.1,builder=builder,
            linear_limit=2,gate_policy='step',model_factory=factory)
        t=h[0]['trials'][h[0]['accepted_trial']]
        self.assertEqual(t['kind'],'cauchy_actual_fallback')
        self.assertEqual(t['seed_model']['gain'],2.)
        self.assertLess(np.linalg.norm(x),np.sqrt(2))

    def test_leave_one_out_preserves_largest_contribution(self):
        b=np.eye(6);r=np.arange(1.,7.);m=HistoryModel(b,b,r,1)
        kept,events=prune_history(m,4,3,4,.1)
        self.assertEqual(kept,[1,2,3]);self.assertEqual(events[0]['removed'],0)

    def test_trigger_counts_fresh_states_not_outer_steps(self):
        def row(fresh,gain):return dict(precondition_rebuilt=fresh,accepted_trial=0,residual=1.,trials=[dict(residual=.9999,seed_model=dict(gain=gain))])
        h=[row(True,1.01),row(False,2.),row(True,1.01),row(False,2.),row(False,2.),row(True,1.01)]
        self.assertTrue(trigger_diagnostic(h)['hypothetical_trigger'])
        h[-1]=row(True,1.2);self.assertFalse(trigger_diagnostic(h)['hypothetical_trigger'])
        self.assertEqual(trigger_diagnostic(h)['low_gain_fresh_streak'],0)

    def test_custom_model_retains_nonlinear_gate(self):
        class F:
            def __call__(self,x):return np.array([10*(x[1]-x[0]**2),1-x[0]])
            def feasible(self,x):return True
        def builder(f,x,r,cache):
            a=np.array([[-20*x[0],10.],[-1.,0.]])
            cache.update(matrix=a,restrict=lambda r:r,lift=lambda g:g,gradient=a.T@r)
            return np.linalg.inv(a),{}
        def factory(x,r,d,jd,h,z,v,derivative,rebuilt,radius,row):
            m=HistoryModel(np.column_stack([z,d]),np.column_stack([v@h,jd]),r,z.shape[1]+1)
            def at_radius(rad):
                s,p,lam=m.full_step([],rad)
                return s,dict(prediction=p,lambda_value=lam,gain=1.)
            return at_radius
        x,h,status=solve_augmented(F(),np.array([-1.2,1.]),max_steps=80,radius=.1,builder=builder,
            gate_policy='step',linear_limit=2,model_factory=factory)
        self.assertEqual(status,'residual_converged')
        np.testing.assert_allclose(x,[1,1],atol=1e-7)
        self.assertTrue(all(r['trials'][r['accepted_trial']]['candidate_model_pass'] for r in h))

if __name__=='__main__':unittest.main()
