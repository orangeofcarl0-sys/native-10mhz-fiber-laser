import unittest
from unittest.mock import patch
import numpy as np
from steady_preconditioner import spectral_coordinates
from steady_support import SUPPORTS
from steady_support_lu import build_factored

class MeritTests(unittest.TestCase):
    def test_full_output_gradient_includes_projection_leakage(self):
        rng=np.random.default_rng(71)
        class Equation:
            n=8;cells=1;dt=.5
            def __init__(self):self.a=rng.normal(size=(35,35))*.2-np.eye(35)
            def __call__(self,x):return self.a@x+np.arange(35)/35
            def batch(self,x):return x@self.a.T+np.arange(35)/35
        f=Equation();x=rng.normal(size=35);r=f(x);audit={}
        restrict,lift,m=spectral_coordinates(8,1,1)
        e=np.column_stack([lift(v) for v in np.eye(m)]);b=f.a@e
        with patch.dict(SUPPORTS,{'tiny':(False,1)}):
            build_factored(f,x,r,'tiny',audit=audit)
        np.testing.assert_allclose(audit['gradient'],b.T@r,rtol=1e-7,atol=1e-7)
        self.assertAlmostEqual(audit['frobenius_norm'],np.linalg.norm(b),places=7)
        self.assertGreater(np.linalg.norm(audit['gradient']-audit['projected_gradient']),.1)
        np.testing.assert_allclose(e.T@e,np.eye(m),atol=1e-14)

    def test_central_coarse_quadratic_exact_tangent(self):
        rng=np.random.default_rng(12)
        class Equation:
            n=8;cells=1;dt=.5
            def __call__(self,x):return x*x+2*x
            def batch(self,x):return x*x+2*x
        f=Equation();x=rng.uniform(.1,.5,35);r=f(x);audit={}
        restrict,lift,m=spectral_coordinates(8,1,1)
        e=np.column_stack([lift(v) for v in np.eye(m)]);b=(2*x+2)[:,None]*e
        with patch.dict(SUPPORTS,{'tiny':(False,1)}):
            inverse,info=build_factored(f,x,r,'tiny',difference='central',epsilon=1e-3,audit=audit)
        np.testing.assert_allclose(audit['gradient'],b.T@r,rtol=1e-10,atol=1e-10)
        v=rng.normal(size=35);c=e.T@v
        expected=e@np.linalg.solve(e.T@b,c)-(v-e@c)
        np.testing.assert_allclose(inverse@v,expected,rtol=1e-10,atol=1e-10)
        self.assertEqual(info['coarse_difference'],'central')

    def test_frobenius_normalization_is_dimension_dependent(self):
        # J=I, nonstationary R=e1: ghat=1/sqrt(d), yet the unit gradient fully descends.
        n=20000;r=np.zeros(n);r[0]=1
        ghat=np.linalg.norm(r)/(np.sqrt(n)*np.linalg.norm(r))
        self.assertLess(ghat,.01)
        self.assertEqual(float(np.dot(r,-r)),-1.)

if __name__=='__main__':unittest.main()
