import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
from gain_age import advance, summary
from config import config, map_initial
from batch_engine import BatchEngine
from spectral_engine import SpectralEngine


class GainAgeTests(unittest.TestCase):
    def test_continuation_stops_without_an_anchor(self):
        import attractor_search
        def unresolved(*args, **kwargs):
            return {'status':'nonstationary_or_unresolved'}, (np.zeros((1,2,8)),np.zeros((1,1)),np.zeros(1)), .5
        with TemporaryDirectory() as directory, patch.object(attractor_search,'ROOT',Path(directory)), patch.object(attractor_search,'run_point',side_effect=unresolved) as runner:
            rows=attractor_search.continuation('CNT_OC',.2,.8,[30,40,50])
            self.assertEqual(runner.call_count,2)
            self.assertTrue(all(r['branch_stopped_without_converged_anchor'] for r in rows))

    def test_frozen_rate_conditional_memory(self):
        rates = np.array([2., 5.])
        age, delta = np.zeros(2), np.ones(2)
        for _ in range(70):
            age = advance(age, rates, .05)
            delta *= np.exp(-rates * .05)
        np.testing.assert_allclose(np.exp(-age), delta, rtol=1e-13)
        self.assertAlmostEqual(summary(age)['gain_age_min'], 7.)

    def test_cellwise_sum_before_minimum_and_resume(self):
        age = advance(np.zeros(2), [1., 10.], 1.)
        resumed = advance(age.copy(), [10., 1.], 1.)
        np.testing.assert_array_equal(resumed, [11., 11.])
        self.assertEqual(summary(resumed)['gain_age_min'], 11.)
        with self.assertRaises(ValueError): advance(age, [1.], 1.)

    def test_rates_match_reference_without_changing_physics(self):
        c = config('CNT_OC', .2)
        a,p,q = map_initial(c, [.8], n=512)
        reference = BatchEngine(c,.5,512,[.05],[.8])
        spectral = SpectralEngine(c,.5,512,[.05],[.8])
        expected=reference.step(a.copy(),p.copy(),q.copy())
        actual=spectral.step(a.copy(),p.copy(),q.copy())
        np.testing.assert_allclose(spectral.rates_b,reference.rates_b,rtol=1e-12)
        for left,right in zip(actual,expected):
            np.testing.assert_allclose(left,right,rtol=1e-10,atol=1e-11)


if __name__ == '__main__': unittest.main()
