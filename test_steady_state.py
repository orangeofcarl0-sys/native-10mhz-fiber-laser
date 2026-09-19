import unittest
import numpy as np
from config import config, map_initial
from spectral_engine import SpectralEngine
from steady_state import CavityResidual, solve


class SteadyTests(unittest.TestCase):
    def test_dynamic_gain_equivalence(self):
        for topology in ("OC_CNT", "CNT_OC"):
            c = config(topology, .2)
            a, p, q = map_initial(c, [.8], n=512, dt=.5)
            residual = CavityResidual(c, .5, a[0], .05, .8)
            x = residual.pack(a[0], p[0])
            r = residual(x)
            dynamic = SpectralEngine(c, .5, 512, [.05], [.8])
            b, out, pp, qq, _ = dynamic.step(a.copy(), p.copy(), q.copy())
            np.testing.assert_allclose(pp[0]-p[0],
                (residual.neq-p[0])*(-np.expm1(-residual.rates_b/dynamic.e.rep)), atol=1e-16)
            m = 2*512
            np.testing.assert_allclose((r[:m]+1j*r[m:2*m]).reshape(2,512),
                                       (b[0]-a[0])/residual.scale, atol=1e-13)
            np.testing.assert_allclose(residual.output, out[0], atol=1e-12)
            np.testing.assert_allclose(qq, c['sa_modulation'], atol=1e-15)
            np.testing.assert_allclose(r[-2:], 0, atol=1e-15)
            self.assertTrue(residual.feasible(x))
            shifted = x.copy(); shifted[-3] = 2
            self.assertFalse(residual.feasible(shifted))

    def test_newton_known_root(self):
        class Equation:
            evaluations = 0
            def __call__(self, x):
                self.evaluations += 1
                return np.array([x[0]**2-2, 3*x[1]-1])
            def feasible(self, x):
                return np.all(x > 0)
        x, history, status = solve(Equation(), np.array([1., 1.]))
        self.assertEqual(status, 'residual_converged')
        np.testing.assert_allclose(x, [np.sqrt(2), 1/3], atol=1e-7)
        self.assertTrue(all(a['residual'] > b['residual'] for a,b in zip(history, history[1:])))

    def test_recovery_guard(self):
        c = config('CNT_OC', .2); c['sa_recovery_ps'] = 1e5
        a, _, _ = map_initial(c, [.8], n=512)
        with self.assertRaises(ValueError):
            CavityResidual(c, .5, a[0], .05, .8)


if __name__ == '__main__':
    unittest.main()
