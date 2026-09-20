import unittest
import numpy as np
from steady_gkb import gkb, svd_trust


class TestGKB(unittest.TestCase):
    def test_rectangular_known_solution(self):
        rng=np.random.default_rng(17)
        a=rng.normal(size=(19,8)); b=rng.normal(size=19)
        u,v,av,B,_=gkb(lambda x:a@x,lambda x:a.T@x,b,8)
        np.testing.assert_allclose(v.T@v,np.eye(8),atol=1e-13)
        np.testing.assert_allclose(u.T@u,np.eye(9),atol=1e-13)
        np.testing.assert_allclose(u@B,av,atol=1e-13)
        for radius in [.01,.5,100]:
            target=np.zeros(len(B));target[0]=np.linalg.norm(b)
            y,p,lam=svd_trust(B,target,radius)
            direct,p0,_=svd_trust(a,b,radius)
            np.testing.assert_allclose(v@y,direct,atol=1e-11)
            self.assertAlmostEqual(p,p0,places=12)
            np.testing.assert_allclose(B.T@(B@y-target)+lam*y,0,atol=1e-11)

    def test_first_column_is_cauchy(self):
        a=np.diag([.1,1,10.]);b=np.array([1.,2.,3.])
        u,v,av,B,_=gkb(lambda x:a@x,lambda x:a.T@x,b,1)
        d=a.T@b;d/=np.linalg.norm(d)
        np.testing.assert_allclose(v[:,0],d)
        target=np.zeros(2);target[0]=np.linalg.norm(b)
        y,_,_=svd_trust(B,target,.3)
        alpha=min(.3,(b@(a@d))/np.linalg.norm(a@d)**2)
        np.testing.assert_allclose(v@y,alpha*d,atol=1e-13)

    def test_rank_deficient_breakdown(self):
        a=np.diag([1.,0.,0.]);b=np.array([1.,1.,0.])
        u,v,av,B,_=gkb(lambda x:a@x,lambda x:a.T@x,b,3)
        self.assertEqual(v.shape[1],1)
        np.testing.assert_allclose(u@B,av,atol=1e-14)


if __name__=='__main__':unittest.main()
