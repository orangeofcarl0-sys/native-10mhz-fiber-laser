import unittest
import numpy as np
from steady_descent_sources import annulus_coordinates,enriched_model,guarded_step,physical_residual,streamed_gradient
from steady_hookstep import arnoldi
from steady_cauchy import augmented_step,cauchy_step

class DescentSourceTests(unittest.TestCase):
    def test_all_parameter_probes_keep_current_pump_and_gauge(self):
        t=np.arange(32)-16;field=np.stack([np.exp(-t*t/20),.1j*np.exp(-t*t/20)])
        state=dict(dt=.5,original_template=field,scale=2.,time_scale=3.,gauge_template=field/2,time_tangent=field*.01)
        for name in ['gdd','oc','psat']:
            f=physical_residual(state,{name:.001})
            self.assertEqual(f.pump,.0275)
            self.assertEqual(f.scale,2.);self.assertEqual(f.time_scale,3.)
            np.testing.assert_array_equal(f.template,state['gauge_template'])
            self.assertAlmostEqual(f.engine.c['segments'][:,0].sum(),20.42)

    def test_streamed_gradient_uses_full_output(self):
        matrix=np.zeros((4,4));matrix[2,0]=3.;matrix[3,1]=2.
        r=np.array([0.,0.,2.,1.]);x=np.zeros(4)
        class F:
            def batch(self,states):return states@matrix.T+r
        lift=lambda g:np.r_[g,0.,0.]
        central,forward=streamed_gradient(F(),x,r,lift,2)
        np.testing.assert_allclose(central,[6.,2.],atol=1e-8)
        np.testing.assert_allclose(forward,[6.,2.],atol=1e-8)

    def test_annulus_orthonormal_and_no_tail(self):
        n=32;cells=2;restrict,lift,size=annulus_coordinates(n,cells,3,7)
        rng=np.random.default_rng(6);g=rng.normal(size=size);x=lift(g)
        np.testing.assert_allclose(restrict(x),g,atol=1e-14)
        self.assertAlmostEqual(np.linalg.norm(x),np.linalg.norm(g))
        np.testing.assert_array_equal(x[4*n:],0)
        spectrum=np.fft.fft((x[:2*n]+1j*x[2*n:4*n]).reshape(2,n),norm='ortho')
        np.testing.assert_allclose(spectrum[:,:4],0,atol=1e-14)
        np.testing.assert_allclose(spectrum[:,-3:],0,atol=1e-14)

    def test_single_direction_matches_existing_augmentation(self):
        rng=np.random.default_rng(4);j=rng.normal(size=(12,12));r=rng.normal(size=12)
        h,z,y,eta,v=arnoldi(lambda x:j@x,r,np.eye(12),limit=3,tolerance=0,return_basis=True)
        d=-j.T@r;d/=np.linalg.norm(d);jd=j@d;radius=.04
        _,pc,_=cauchy_step(r,d,jd,radius,np.ones(12))
        expected,guard,_=augmented_step(h,z,v,r,d,jd,radius,np.ones(12))
        step,actual=guarded_step(enriched_model(h,z,v,r,d[:,None],jd[:,None]),r,radius,pc)
        np.testing.assert_allclose(step,expected,atol=1e-10)
        self.assertAlmostEqual(actual['prediction'],guard['selected_prediction'],places=12)

    def test_extra_parameter_uses_rectangular_output_without_added_equation(self):
        r=np.array([1.,2.,3.,4.]);j=np.diag([1.,2.,3.,4.])
        h,z,y,eta,v=arnoldi(lambda x:j@x,r,np.eye(4),limit=1,tolerance=0,return_basis=True)
        d=-j.T@r;d/=np.linalg.norm(d);jd=j@d
        zpad=np.vstack([z,np.zeros(z.shape[1])]);dpad=np.r_[d,0];p=np.array([0.,0,0,0,1.])
        parameter=-r;model=enriched_model(h,zpad,v,r,np.column_stack([dpad,p]),np.column_stack([jd,parameter]))
        _,pc,_=cauchy_step(r,d,jd,.1,np.ones(4));step,info=guarded_step(model,r,.1,pc)
        response=np.column_stack([j,parameter])@step
        self.assertAlmostEqual(info['prediction'],float(-r@response-.5*response@response),places=12)
        self.assertTrue(info['raw_model_pass']);self.assertLessEqual(np.linalg.norm(step),.10000001)
        self.assertGreater(abs(step[-1]),0)

if __name__=='__main__':unittest.main()
