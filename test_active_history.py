import unittest
import numpy as np
from steady_active_history import HistoryModel, greedy_history
from steady_descent_sources import enriched_model, guarded_step


class ActiveHistoryTests(unittest.TestCase):
    def test_compression_matches_full_metric(self):
        rng=np.random.default_rng(4);r=rng.normal(size=35)
        z=rng.normal(size=(35,4));v,_=np.linalg.qr(rng.normal(size=(35,5)))
        h=rng.normal(size=(5,4));d=rng.normal(size=(35,5));j=rng.normal(size=(35,5))
        m=HistoryModel(np.column_stack([z,d]),np.column_stack([v@h,j]),r,5)
        for ids in [[],[0,2],[0,1,2,3]]:
            use=[0]+[i+1 for i in ids]
            full=enriched_model(h,z,v,r,d[:,use],j[:,use])
            for radius in [.01,1.,10.]:
                a,p,_=m.full_step(ids,radius);b,info=guarded_step(full,r,radius,0)
                np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-10)
                self.assertAlmostEqual(p,info['prediction'],places=10)

    def test_uphill_direction_and_forced_ladder(self):
        r=np.array([1.,2.,3.,4.,5.]);b=np.eye(5)
        m=HistoryModel(b,b,r,1)
        result=greedy_history(m,4,1.,stop_gain=100.)
        self.assertEqual(result['ladder'][0]['selected'],[3])
        self.assertEqual(result['policy_count'],0)
        self.assertEqual(len(result['ladder']),4)
        previous=m.solve([],1.)[1]
        for row in result['ladder']:
            self.assertEqual(row['prediction'],max(v['prediction'] for v in row['candidates']))
            self.assertGreaterEqual(row['prediction']+1e-12,previous)
            previous=row['prediction']

    def test_near_duplicate_and_deterministic_selection(self):
        rng=np.random.default_rng(8);b=rng.normal(size=(20,6));b[:,5]=b[:,4]+1e-11*b[:,3]
        j=rng.normal(size=(20,20))@b;r=rng.normal(size=20)
        m=HistoryModel(b,j,r,2)
        a=greedy_history(m,4,.1);c=greedy_history(m,4,.1)
        self.assertEqual(a,c)
        step,_,_=m.full_step(a['ladder'][-1]['selected'],.1)
        self.assertLessEqual(np.linalg.norm(step),.100001)


if __name__=='__main__':unittest.main()
