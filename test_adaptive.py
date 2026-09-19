"""Known-answer and cross-implementation tests for the P0 solver."""

import unittest
import numpy as np
from scipy.fft import fft, ifft
from config import config, map_initial
from batch_engine import BatchEngine
from adaptive_solver import (
    AdaptiveSolver,
    AdaptiveEngine,
    DEFAULTS,
    regrid,
    recenter,
    remap_population,
    pump_initial,
    NumericalLimit,
)
from attractor_search import classify, field_distance, accelerated_gain


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.c = config("OC_CNT", 0.2)
        self.a, self.pop, self.q = map_initial(self.c, [0.3], 1024, 0.5)

    def test_interpolation_padding_translation_conserve_energy(self):
        shifted = np.roll(self.a, 450, axis=-1)
        a, _ = recenter(shifted, 0.5)
        a, dt = regrid(a, 0.5, True, True)
        self.assertEqual(a.shape[-1], 4096)
        self.assertAlmostEqual(
            np.sum(abs(a) ** 2) * dt, np.sum(abs(self.a) ** 2) * 0.5, places=10
        )

    def test_population_remap_preserves_integral(self):
        pop = np.array([np.linspace(0.1, 0.8, 15)])
        fine = remap_population(pop, 60)
        np.testing.assert_allclose(remap_population(fine, 15), pop, rtol=1e-13)

    def test_pump_conditioning(self):
        dark = pump_initial(self.c, 0.0, 0.3, n=1024)[1]
        low = pump_initial(self.c, 0.005, 0.3, n=1024)[1]
        high = pump_initial(self.c, 0.05, 0.3, n=1024)[1]
        np.testing.assert_allclose(dark, 0, atol=1e-14)
        self.assertTrue(np.all(high > low))
        self.assertTrue(np.all(np.diff(high[0]) < 0))

    def test_gain_search_update_matches_exact_relaxation(self):
        pop = np.array([[0.4999, 0.5001]])
        B = np.full_like(pop, 1000.0)
        A = 0.5 * B
        advanced, info = accelerated_gain(pop, A, B, 1e7)
        expected = 0.5 + (pop - 0.5) * np.exp(
            -B * info["nominal_round_multiplier"] / 1e7
        )
        np.testing.assert_allclose(advanced, expected, rtol=1e-14)
        self.assertLess(np.max(abs(advanced - pop)), 0.002)

    def test_population_drift_cannot_cancel_in_spatial_mean(self):
        fields = [self.a[0].copy() for _ in range(80)]
        populations = []
        direction = np.r_[np.ones(7), 0.0, -np.ones(7)]
        for k in range(80):
            populations.append(self.pop[0] + k * 1e-4 * direction)
        self.assertEqual(
            classify(fields, populations, 0.5, 6.0)["status"],
            "nonstationary_or_unresolved",
        )

    def test_frozen_gain_does_not_age_population(self):
        c = self.c.copy()
        c["gain_mode"] = "frozen"
        e = BatchEngine(c, 0.5, 1024, np.array([0.02]), np.array([0.3]))
        _, _, pop, _, _ = e.step(self.a.copy(), self.pop.copy(), self.q.copy())
        np.testing.assert_array_equal(pop, self.pop)
        self.assertTrue(np.all(e.rates_b > 0))

    def test_passive_step_doubling_against_fine_reference(self):
        c = self.c.copy()
        c["passive_step_m"] = 0.02
        reference = BatchEngine(c, 0.5, 1024, np.array([0.02]), np.array([0.3]))
        expected = reference.fiber(self.a.copy(), 5, np.array([0.02]), self.pop.copy())[
            0
        ]
        adaptive = AdaptiveEngine(self.c, 0.5, 1024, 0.02, 0.3, DEFAULTS)
        actual = adaptive.fiber(self.a.copy(), 5, np.array([0.02]), self.pop.copy())[0]
        self.assertLess(
            np.linalg.norm(actual - expected) / np.linalg.norm(expected), 2e-5
        )
        self.assertAlmostEqual(
            np.sum(abs(actual) ** 2) / np.sum(abs(self.a) ** 2), 1.0, places=11
        )

    def test_no_false_certification_from_short_gain_age(self):
        fields = [self.a[0].copy() for _ in range(80)]
        populations = [self.pop[0].copy() for _ in range(80)]
        self.assertEqual(
            classify(fields, populations, 0.5, 0.1)["status"], "provisional_recurrence"
        )
        self.assertEqual(classify(fields, populations, 0.5, 7.0)["period"], 1)

    def test_period_two_and_polarization_changes(self):
        x = self.a[0]
        y = x[::-1].copy()
        self.assertLess(field_distance(x, y)[0], 1e-12)
        self.assertGreater(field_distance(x, y)[1], 1.0)
        fields = [x.copy() if i % 2 == 0 else y.copy() for i in range(80)]
        result = classify(fields, [self.pop[0].copy() for _ in fields], 0.5, 6.0)
        self.assertEqual(result["period"], 2)

    def test_rollback_does_not_double_advance_gain(self):
        # Force one warning after a fully calculated trial; rerun from original state.
        options = dict(edf_rtol=1.0, edf_population_atol=1.0)
        solver = AdaptiveSolver(self.c, 0.02, 0.3, 0.5, 1024, options)
        original = solver.engine.step

        def trigger(*args, **kwargs):
            result = original(*args, **kwargs)
            solver.engine.worst_edges[1] = 1e-9
            return result

        solver.engine.step = trigger
        result = solver.step(self.a, self.pop, self.q)
        self.assertEqual(solver.round, 1)
        self.assertEqual(len(solver.events), 1)
        refined, dt = regrid(self.a, 0.5, True, False)
        clean = AdaptiveSolver(self.c, 0.02, 0.3, dt, refined.shape[-1], options)
        expected = clean.step(refined, self.pop, self.q)
        for left, right in zip(result, expected):
            np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-12)

    def test_resource_limit_leaves_input_unchanged(self):
        a = self.a.copy()
        pop = self.pop.copy()
        q = self.q.copy()
        solver = AdaptiveSolver(
            self.c, 0.02, 0.3, 0.5, 1024, dict(time_warning=1e-40, max_samples=1024)
        )
        with self.assertRaises(NumericalLimit):
            solver.step(a, pop, q)
        np.testing.assert_array_equal(a, self.a)
        np.testing.assert_array_equal(pop, self.pop)
        self.assertEqual(solver.round, 0)

    def test_edf_replay_matches_finer_start(self):
        solver = AdaptiveSolver(self.c, 0.02, 0.3, 0.5, 1024)
        actual = solver.step(self.a, self.pop, self.q)
        self.assertTrue(
            any(e["action"] == "rollback_edf_refine" for e in solver.events)
        )
        direct = AdaptiveSolver(solver.c, 0.02, 0.3, 0.5, 1024)
        pop = remap_population(self.pop, direct.engine.e.ops[1][0])
        expected = direct.step(self.a, pop, self.q)
        for left, right in zip(actual, expected):
            np.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
