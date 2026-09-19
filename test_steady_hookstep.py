import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import aslinearoperator
from config import config,map_initial
from steady_state import CavityResidual
from steady_hookstep import arnoldi,hookstep,solve_globalized
from steady_diagnostics import reanchor


class HookstepTests(unittest.TestCase):
    def test_actual_preconditioned_norm(self):
        h=np.array([[1.,0],[0,1],[0,0]])
        z=np.diag([100.,.01])
        step,prediction,_=hookstep(h,z,1.,1.,np.ones(2))
        np.testing.assert_allclose(step,[1,0],atol=1e-10)
        self.assertAlmostEqual(prediction,.5*(1-.99**2),places=10)

    def test_weighted_constraint_matches_independent_optimizer(self):
        h=np.array([[2.,.3],[.1,1.],[.3,-.2]])
        z=np.array([[2.,1.],[.1,3.],[1.,-.2],[.3,.5]])
        weights=np.array([1.,2.,.5,3.]);radius=.2
        target=np.array([1.,0,0])
        objective=lambda y:.5*np.linalg.norm(target-h@y)**2
        answer=minimize(objective,np.zeros(2),method='SLSQP',constraints=[
            dict(type='ineq',fun=lambda y:radius**2-np.linalg.norm(weights*(z@y))**2)],
            options=dict(ftol=1e-13,maxiter=200))
        step,_,_=hookstep(h,z,1.,radius,weights)
        self.assertTrue(answer.success)
        np.testing.assert_allclose(step,z@answer.x,atol=1e-7)
        self.assertLessEqual(np.linalg.norm(weights*step),radius*(1+1e-9))

    def test_right_arnoldi_linear_solution(self):
        matrix=np.array([[3.,1.],[.2,2.]])
        r=np.array([1.,-2.]);preconditioner=aslinearoperator(np.diag([2.,.1]))
        h,z,y,relative=arnoldi(lambda v:matrix@v,r,preconditioner,limit=2,tolerance=1e-10)
        np.testing.assert_allclose(z@y,np.linalg.solve(matrix,-r),atol=1e-12)
        self.assertLess(relative,1e-10)

    def test_hookstep_rosenbrock(self):
        class Equation:
            evaluations=0
            def __call__(self,x):
                self.evaluations+=1
                return np.array([10*(x[1]-x[0]**2),1-x[0]])
            def feasible(self,x):return np.isfinite(x).all()
        with patch('steady_hookstep.build_coarse',return_value=(aslinearoperator(np.eye(2)),{})), \
             patch('steady_hookstep.gauge_geometry',return_value={'condition':1}), \
             patch('steady_hookstep.blocks',side_effect=lambda _,s:{'total':float(np.linalg.norm(s))}):
            x,history,status=solve_globalized(Equation(),np.array([-1.2,1.]),max_steps=80)
        self.assertEqual(status,'residual_converged')
        np.testing.assert_allclose(x,[1,1],atol=1e-7)
        self.assertTrue(any(len(row.get('trials',[]))>1 for row in history))

    def test_reanchor_preserves_physical_equations_and_units(self):
        c=config('CNT_OC',.2);a,p,_=map_initial(c,[.8],n=256)
        residual=CavityResidual(c,.5,a[0],.05,.8)
        x=residual.pack(a[0]*1.1*np.exp(.3j),p[0],phase=.4,shift_ps=.2)
        before=residual(x);scale=residual.scale;time_scale=residual.time_scale
        copy=x.copy();reanchor(residual,x);after=residual(x)
        np.testing.assert_allclose(after[:-2],before[:-2],atol=1e-14)
        np.testing.assert_allclose(after[-2:],0,atol=1e-14)
        np.testing.assert_array_equal(x,copy)
        self.assertEqual(residual.scale,scale);self.assertEqual(residual.time_scale,time_scale)


if __name__=='__main__':unittest.main()
