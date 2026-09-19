import unittest
import numpy as np
from config import config, map_initial
from steady_state import CavityResidual, solve
from steady_preconditioner import optical_transfer, schur_inverse, build, spectral_coordinates, build_coarse


class PreconditionerTests(unittest.TestCase):
    def test_batched_probes_match_single_maps(self):
        c=config('CNT_OC',.2)
        a,p,_=map_initial(c,[.8],n=256)
        residual=CavityResidual(c,.5,a[0],.05,.8)
        x=residual.pack(a[0],p[0],phase=.3,shift_ps=.2)
        probes=x+np.random.default_rng(3).normal(size=(3,len(x)))*1e-6
        expected=np.array([residual(v) for v in probes])
        actual=residual.batch(probes)
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-10)

    def test_fourier_orthonormal_projection(self):
        restrict,lift,size = spectral_coordinates(64,3,5)
        rng=np.random.default_rng(10)
        v=rng.normal(size=size);x=rng.normal(size=4*64+5)
        np.testing.assert_allclose(restrict(lift(v)),v,atol=1e-14)
        self.assertAlmostEqual(np.dot(lift(v),x),np.dot(v,restrict(x)),places=12)
        self.assertAlmostEqual(np.linalg.norm(lift(v)),np.linalg.norm(v),places=12)

    def test_coarse_exact_subspace_limit(self):
        restrict,lift,size = spectral_coordinates(16,1,3)
        q=np.column_stack([lift(v) for v in np.eye(size)])
        matrix=-np.eye(67)+q@np.diag(np.linspace(2,3,size))@q.T
        class Equation:
            n=16
            cells=1
            def __call__(self,x):
                return matrix@x
        x=np.random.default_rng(8).normal(size=67)
        equation=Equation()
        operator,_=build_coarse(equation,x,equation(x),cutoff=3)
        np.testing.assert_allclose(operator@(matrix@x),x,atol=1e-7)

    def test_right_preconditioned_newton_known_solution(self):
        restrict,lift,size=spectral_coordinates(16,1,3)
        q=np.column_stack([lift(v) for v in np.eye(size)])
        matrix=-np.eye(67)+q@np.diag(np.linspace(2,3,size))@q.T
        target=np.random.default_rng(12).normal(size=67)
        class Equation:
            n=16
            cells=1
            evaluations=0
            def __call__(self,x):
                self.evaluations+=1
                return matrix@(x-target)
            def feasible(self,x):
                return np.isfinite(x).all()
        result,_,status=solve(Equation(),np.zeros(67),precondition='coarse',
                              coarse_cutoff=3,rebuild_every=3,max_steps=4)
        self.assertEqual(status,'residual_converged')
        np.testing.assert_allclose(result,target,atol=1e-7)

    def test_linear_cavity_limit(self):
        for topology in ('OC_CNT', 'CNT_OC'):
            c = config(topology, .2)
            c['segments'][:, 3] = 0  # No Kerr response.
            c['sa_modulation'] = 0  # Constant CNT transmission.
            c['segments'][:, 4] = .1
            c['segments'][:, 5] = np.linspace(.1, .6, 6)
            c['segments'][:, 6] = .02
            a, p, _ = map_initial(c, [.8], n=256)
            a[0, 1] = a[0, 0]*(.2+.3j)
            residual = CavityResidual(c, .5, a[0], .05, .8)
            x = residual.pack(a[0], p[0], phase=.2, shift_ps=.15)
            r = residual(x)
            actual = a[0]/residual.scale+(r[:512]+1j*r[512:1024]).reshape(2,256)
            expected = np.fft.ifft(np.einsum('kij,jk->ik', optical_transfer(residual,x),
                                            np.fft.fft(a[0]/residual.scale)))
            np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-11)

    def test_schur_dense_reference(self):
        rng = np.random.default_rng(8)
        a = np.diag(np.arange(1, 9.))
        u = rng.normal(size=(8, 3)); v = rng.normal(size=(3, 8))
        d = 10*np.eye(3)
        preconditioner, _ = schur_inverse(lambda z:np.linalg.solve(a,z),u,v,d)
        matrix = np.block([[a,u],[v,d]])
        b = rng.normal(size=11)
        np.testing.assert_allclose(preconditioner@b,np.linalg.solve(matrix,b),atol=1e-13)

    def test_build_does_not_modify_state_or_equation(self):
        c = config('CNT_OC', .2)
        a,p,_ = map_initial(c,[.8],n=256)
        residual = CavityResidual(c,.5,a[0],.05,.8)
        x = residual.pack(a[0],p[0]); original=x.copy()
        r = residual(x)
        operator, diagnostics = build(residual,x,r)
        self.assertTrue(np.isfinite(operator@r).all())
        self.assertGreater(diagnostics['schur_condition'], 0)
        np.testing.assert_array_equal(x,original)
        np.testing.assert_allclose(residual(x),r,atol=1e-14)


if __name__ == '__main__':
    unittest.main()
