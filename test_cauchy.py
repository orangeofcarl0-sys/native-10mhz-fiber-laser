import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse.linalg import aslinearoperator
from steady_hookstep import arnoldi,hookstep
from steady_cauchy import span_audit,cauchy_step,augmented_model,augmented_step

class CauchyTests(unittest.TestCase):
    def test_arnoldi_output_basis_preserves_relation(self):
        a=np.array([[2.,4.,.1],[0.,1.,3.],[1.,0.,5.] ]);r=np.array([1.,2.,3.])
        h,z,y,eta,v=arnoldi(lambda x:a@x,r,aslinearoperator(np.eye(3)),limit=2,tolerance=0,return_basis=True)
        np.testing.assert_allclose(a@z,v@h,atol=1e-13)
        np.testing.assert_allclose(v[:,0],-r/np.linalg.norm(r))
        self.assertEqual(len(arnoldi(lambda x:a@x,r,aslinearoperator(np.eye(3)),limit=2)),4)

    def test_span_distinguishes_metric_truncation(self):
        rows,_,_=span_audit(np.diag([1.,1e-13]),np.array([0.,1.]),np.ones(2))
        self.assertLess(rows[0]['epsilon'],1e-12);self.assertEqual(rows[1]['epsilon'],1.)
        for row in rows:self.assertAlmostEqual(row['partition'],1.)

    def test_augmentation_contains_cauchy_in_exact_nonsymmetric_model(self):
        a=np.array([[1.,20.,0],[0,2.,3.],[1.,0.,1.] ]);r=np.array([1.,.3,-.2])
        h,z,_,_,v=arnoldi(lambda x:a@x,r,aslinearoperator(np.eye(3)),limit=1,tolerance=0,return_basis=True)
        d=-a.T@r;d/=np.linalg.norm(d);jd=a@d;weights=np.array([1.,2.,.5])
        for radius in [.001,.01,.1]:
            step,info,raw=augmented_step(h,z,v,r,d,jd,radius,weights)
            self.assertTrue(info['raw_model_pass']);self.assertFalse(info['fallback'])
            exact=.5*(np.dot(r,r)-np.linalg.norm(r+a@raw)**2)
            self.assertAlmostEqual(exact,info['raw_prediction'],places=12)
            self.assertGreaterEqual(exact,info['cauchy_prediction']*(1-1e-6))
            self.assertLessEqual(np.linalg.norm(weights*step),radius*(1+1e-7))

    def test_runtime_fallback_does_not_hide_optimizer_failure(self):
        r=np.array([-1.,0.]);d=np.array([1.,0.]);jd=d.copy()
        with patch('steady_cauchy.hookstep',return_value=(np.zeros(2),0.,0.)):
            step,info,_=augmented_step(np.eye(2),np.eye(2),np.eye(2),r,d,jd,.1,np.ones(2))
        self.assertFalse(info['raw_model_pass']);self.assertTrue(info['fallback'])
        np.testing.assert_allclose(step,[.1,0.])

    def test_original_reduced_optimizer_beats_feasible_cauchy(self):
        a=np.array([[1.,4.,0],[0,2.,3.],[1.,0.,1.] ]);r=np.array([1.,.3,-.2])
        h,z,_,_,v=arnoldi(lambda x:a@x,r,aslinearoperator(np.eye(3)),limit=3,tolerance=0,return_basis=True)
        d=-a.T@r;d/=np.linalg.norm(d);weights=np.array([1.,2.,.5])
        for radius in [.001,.01,.1]:
            sc,pc,_=cauchy_step(r,d,a@d,radius,weights)
            yc=np.linalg.lstsq(z,sc,rcond=1e-14)[0]
            target=np.r_[np.linalg.norm(r),np.zeros(h.shape[0]-1)]
            np.testing.assert_allclose(z@yc,sc,atol=1e-13)
            self.assertAlmostEqual(np.linalg.norm(target-h@yc),np.linalg.norm(r+a@sc),places=12)
            sh,ph,_=hookstep(h,z,np.linalg.norm(r),radius,weights)
            self.assertGreaterEqual(ph,pc*(1-1e-6))

if __name__=='__main__':unittest.main()
