import unittest
import numpy as np
from steady_gkb_recycling import orthogonal_union
from steady_gkb import svd_trust


class RecyclingTests(unittest.TestCase):
    def test_duplicate_union_metric_and_response(self):
        rng=np.random.default_rng(6)
        q,_=np.linalg.qr(rng.normal(size=(17,8)));a=rng.normal(size=(21,17))
        basis=np.column_stack([q[:,:6],q[:,3:]])
        u,ju,info,transform=orthogonal_union(basis,a@basis)
        self.assertEqual(info['rank'],8)
        np.testing.assert_allclose(u.T@u,np.eye(8),atol=1e-13)
        np.testing.assert_allclose(ju,a@u,atol=1e-13)
        np.testing.assert_allclose(basis@transform,u,atol=1e-13)
        b=rng.normal(size=21)
        for radius in [.01,1,100]:
            y,p,_=svd_trust(ju,b,radius);z,p0,_=svd_trust(a@q,b,radius)
            np.testing.assert_allclose(u@y,q@z,atol=1e-11)
            self.assertAlmostEqual(p,p0,places=12)


if __name__=='__main__':unittest.main()
