import unittest
import numpy as np
from steady_active_history import HistoryModel
from steady_history_complete import exhaustive_history,paired_novelty

class CompleteHistoryTests(unittest.TestCase):
    def test_exhaustive_equal_cardinality(self):
        b=np.eye(5);r=np.arange(1.,6.);m=HistoryModel(b,b,r,1)
        audit=exhaustive_history(m,4,.2)
        self.assertEqual(len(audit['subsets']),16)
        self.assertEqual(audit['best_by_count'][2]['selected'],[2,3])
        self.assertTrue(all(audit['best_by_count'][k]['prediction']<=audit['best_by_count'][k+1]['prediction']+1e-12 for k in range(4)))

    def test_separate_span_membership_is_not_paired_membership(self):
        b=np.array([[1.],[0.]])
        result=paired_novelty(b,b,np.array([1.,0.]),np.array([-1.,0.]))
        self.assertLess(result['state']['relative'],1e-14)
        self.assertLess(result['response']['relative'],1e-14)
        self.assertGreater(result['joint']['relative'],.9)

    def test_known_projection_and_duplicate(self):
        b=np.array([[1.,1.],[0.,0.],[0.,0.]])
        d=np.array([1.,1.,0.]);a=paired_novelty(b,2*b,d,2*d)
        self.assertAlmostEqual(a['state']['relative'],1/np.sqrt(2))
        self.assertEqual(a['state']['rank'],1)
        self.assertAlmostEqual(a['joint']['relative'],1/np.sqrt(2))

if __name__=='__main__':unittest.main()
