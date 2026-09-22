import unittest
import numpy as np
from local_geometry import curvature_probe

class TestGeometry(unittest.TestCase):
    def test_known_indefinite_objective_hessian(self):
        x=np.array([.4,.3]);d=np.array([1.,0.])
        def vg(z):
            r=np.array([z[0]**2-1,2*z[1]]);J=np.diag([2*z[0],2]);return r,J.T@r
        J=np.diag([.8,2.]);r,_=vg(x)
        rows,vec,error=curvature_probe(vg,x,r,d,lambda v:J@v,lambda v:J.T@v)
        self.assertLess(error,1e-12)
        self.assertAlmostEqual(rows[-1]['full'],-1.04,places=7)
        self.assertAlmostEqual(rows[-1]['missing'],-1.68,places=7)
        self.assertAlmostEqual(rows[-1]['kappa'],2.625,places=7)
        self.assertLess(rows[-1]['full'],0)
    def test_small_chi_does_not_imply_inconsistent_linear_system(self):
        J=np.diag([1.,1e-6]);r=np.array([0.,1.]);g=J.T@r
        chi=np.linalg.norm(g)/(np.linalg.norm(J,2)*np.linalg.norm(r))
        s=np.linalg.solve(J,-r)
        self.assertLess(chi,1e-4)
        self.assertLess(np.linalg.norm(r+J@s),1e-14)
        def vg(x):
            value=r+J@x;return value,J.T@value
        rows,_,_=curvature_probe(vg,np.zeros(2),r,np.array([1.,0.]),lambda v:J@v,lambda v:J.T@v)
        self.assertLess(max(t['kappa'] for t in rows),1e-8)
if __name__=='__main__':unittest.main()
