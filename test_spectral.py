import unittest
import numpy as np
from config import config, map_initial
from batch_engine import BatchEngine
from spectral_engine import SpectralEngine


class SpectralTests(unittest.TestCase):
    def test_round_map_and_population(self):
        for topology in ("OC_CNT", "CNT_OC"):
            for gdd in (-0.1, 0.2):
                c = config(topology, gdd)
                # Exercise nontrivial rotations, birefringence, DGD and both polarizations.
                c["segments"][:, 4] = 0.12
                c["segments"][:, 5] = np.linspace(0.1, 0.6, 6)
                c["segments"][:, 6] = 0.02
                oc = np.array([0.2, 0.5, 0.9])
                pumps = np.array([0.005, 0.02, 0.15])
                a, p, q = map_initial(c, oc, n=512)
                a[:, 1] = a[:, 0] * (0.2 + 0.1j)
                b, pb, qb = a.copy(), p.copy(), q.copy()
                reference = BatchEngine(c, 0.5, 512, pumps, oc)
                spectral = SpectralEngine(c, 0.5, 512, pumps, oc)
                for _ in range(4):
                    a, out, p, q, sa = reference.step(a, p, q)
                    b, ob, pb, qb, sb = spectral.step(b, pb, qb)
                    self.assertLess(np.linalg.norm(a - b) / np.linalg.norm(a), 1e-10)
                    np.testing.assert_allclose(ob, out, rtol=1e-10, atol=1e-11)
                    np.testing.assert_allclose(pb, p, atol=1e-13)
                    np.testing.assert_allclose(sb, sa, rtol=1e-10, atol=1e-12)
                expected = 2 * sum(x[0] for x in spectral.ops) + 4
                self.assertEqual(spectral.transform_count, 4 * expected)

    def test_spectral_input_and_frozen_population(self):
        c = config("OC_CNT", 0.2)
        c["gain_mode"] = "frozen"
        a, p, q = map_initial(c, [0.3], n=512)
        engine = SpectralEngine(c, 0.5, 512, [0.02], [0.3])
        f, out, po, qo, sa = engine.step_spectrum(np.fft.fft(a), p.copy(), q.copy())
        time = engine.step(a, p.copy(), q.copy())
        np.testing.assert_allclose(np.fft.ifft(f), time[0], atol=1e-12)
        np.testing.assert_array_equal(po, p)


if __name__ == "__main__":
    unittest.main()
