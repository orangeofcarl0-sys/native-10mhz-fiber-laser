import unittest
import numpy as np
from config import config, map_initial
from batch_engine import BatchEngine
from shared_gain import SharedGainEngine, satellite_windows, integrated_stokes
from shared_gain import close_satellite
from floquet import multipliers


class SharedGainTests(unittest.TestCase):
    def test_two_windows_consume_sum_power(self):
        c = config("OC_CNT", 0.2)
        a, p, q = map_initial(c, [0.3], n=512)
        single = BatchEngine(c, 0.5, 512, [0.02], [0.3])
        shared = SharedGainEngine(c, 0.5, 512, 0.02, 0.3)
        # Linear propagation isolates total-power gain depletion from Kerr effects.
        for engine in (single, shared):
            op = engine.e.ops[1]
            engine.e.ops[1] = (*op[:-1], 0.0)
        reference = single.fiber(np.sqrt(2) * a, 1, np.array([0.02]), p.copy())
        result = shared.fiber(np.repeat(a, 2, axis=0), 1, np.array([0.02]), p.copy())
        np.testing.assert_allclose(result[2], reference[2], atol=1e-14)
        np.testing.assert_allclose(
            result[0][0] * np.sqrt(2), reference[0][0], atol=1e-12
        )

    def test_floquet_known_map_and_gate(self):
        diagonal = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        neutral = np.eye(8)[:, :1]
        values = multipliers(lambda x: diagonal * x, np.zeros(8), neutral, count=2)
        np.testing.assert_allclose(sorted(abs(values)), [0.8, 0.9], atol=1e-8)
        with self.assertRaises(ValueError):
            multipliers(lambda x: diagonal * x, np.ones(8), neutral, count=2)

    def test_close_zero_perturbation(self):
        a = np.ones((2, 128), complex)
        np.testing.assert_array_equal(close_satellite(a, 0.5, 20, 0), a)

    def test_zero_satellite_and_single_window(self):
        for topology in ("OC_CNT", "CNT_OC"):
            c = config(topology, 0.2)
            a, p, q = map_initial(c, [0.3], n=512)
            reference = BatchEngine(c, 0.5, 512, [0.02], [0.3]).step(
                a.copy(), p.copy(), q.copy()
            )
            for windows in (1, 2):
                field = a.copy() if windows == 1 else satellite_windows(a[0], 0)
                result = SharedGainEngine(c, 0.5, 512, 0.02, 0.3, windows).step(
                    field, p.copy(), np.repeat(q, windows)
                )
                np.testing.assert_allclose(
                    result[0][0], reference[0][0], rtol=1e-11, atol=1e-12
                )
                np.testing.assert_allclose(result[2], reference[2], atol=1e-14)

    def test_window_permutation(self):
        c = config("OC_CNT", 0.2)
        a, p, q = map_initial(c, [0.3], n=512)
        field = satellite_windows(a[0], 0.01)
        e = SharedGainEngine(c, 0.5, 512, 0.02, 0.3)
        r1 = e.step(field.copy(), p.copy(), np.repeat(q, 2))
        r2 = e.step(field[::-1].copy(), p.copy(), np.repeat(q, 2))
        np.testing.assert_allclose(r1[0], r2[0][::-1], atol=1e-12)
        np.testing.assert_allclose(r1[2], r2[2], atol=1e-14)

    def test_stokes_global_phase_invariant(self):
        a = np.array([[1, 2], [1j, 2j]])
        np.testing.assert_allclose(integrated_stokes(a), [0, 0, -1])
        np.testing.assert_allclose(
            integrated_stokes(a * np.exp(0.3j)), integrated_stokes(a), atol=1e-15
        )


if __name__ == "__main__":
    unittest.main()
