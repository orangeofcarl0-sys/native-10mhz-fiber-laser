import unittest
import numpy as np
from steady_window import embed,center,EmbeddedResidual
from steady_hookstep import progress_window
from steady_ptc import mass_sign,shifted_local_inverse
from steady_preconditioner import spectral_coordinates


class WindowTests(unittest.TestCase):
    def test_shifted_local_inverse_known_block_operator(self):
        n=8;cells=3;mu=.1
        restrict,lift,size=spectral_coordinates(n,cells,1)
        matrix=np.diag(np.r_[-np.arange(1,size-cells-1),[2.,3.,4.,1.,1.]])
        self.assertEqual(matrix.shape,(size,size))
        rng=np.random.default_rng(21);v=rng.normal(size=8*n+cells+2)
        coarse=restrict(center(v,n));projected=embed(lift(coarse),n)
        response=embed(lift(matrix@coarse),n)-(v-projected)+mu*mass_sign(2*n,cells)*v
        inverse=shifted_local_inverse(matrix,n,cells,mu,cutoff=1)
        np.testing.assert_allclose(inverse@response,v,atol=1e-12)

    def test_ptc_field_population_and_gauge_signs(self):
        sign=mass_sign(2,3)
        np.testing.assert_array_equal(sign,np.r_[-np.ones(8),np.ones(3),[0,0]])
        # Stable contraction Phi(a)=0.5a: correct sign damps, +mu I grows.
        a=1.;r=-.5*a;j=-.5;mu=1.
        self.assertLess(abs(a-r/(j-mu)),abs(a))
        self.assertGreater(abs(a-r/(j+mu)),abs(a))

    def test_embedding_isometry_and_adjoint(self):
        rng=np.random.default_rng(19);n=32
        x=rng.normal(size=4*n+5);y=rng.normal(size=8*n+5)
        np.testing.assert_array_equal(center(embed(x,n),n),x)
        self.assertAlmostEqual(np.linalg.norm(embed(x,n)),np.linalg.norm(x))
        self.assertAlmostEqual(np.dot(embed(x,n),y),np.dot(x,center(y,n)))

    def test_localized_map_retains_outer_base_state(self):
        class Wide:
            cells=3
            def __call__(self,x):return x*x
            def batch(self,x):return x*x
        rng=np.random.default_rng(20);n=8;base=rng.normal(size=8*n+5)
        local=EmbeddedResidual(Wide(),n,base)
        core=center(base,n)
        np.testing.assert_allclose(local(core),center(base*base,n))
        np.testing.assert_allclose(local.batch(np.array([core,core+.01])),
                                   [local(core),local(core+.01)])

    def test_progress_stop_needs_both_conditions(self):
        history=[dict(residual=1-.001*i,accepted_trial=0,trials=[{'rho':.8}]) for i in range(6)]
        self.assertTrue(progress_window(history)['stop'])
        for h in history:h['trials'][0]['rho']=.2
        self.assertFalse(progress_window(history)['stop'])
        for i,h in enumerate(history):h['residual']=.8**i
        self.assertEqual(progress_window(history)['label'],'progress')


if __name__=='__main__':unittest.main()
