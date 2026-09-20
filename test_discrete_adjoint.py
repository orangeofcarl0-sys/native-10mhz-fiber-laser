"""Layer A-E checks independent of Krylov or trust-step behavior."""
import unittest
from types import SimpleNamespace
import numpy as np
from config import config
from steady_state import CavityResidual
from discrete_adjoint import DiscreteAdjoint
from adjoint_primitives import shell_value,shell_vjp,kerr_value,kerr_vjp,cnt_value,cnt_vjp
from batch_engine import kerr,absorber

SCALES=[1e-4,3e-5,1e-5,3e-6,1e-6]


def inner(x,y): return float(np.vdot(x,y).real)


def dot_scan(function,reverse,x,u,v):
    right=inner(u,reverse(x,v))
    rows=[]
    for h in SCALES:
        left=inner((function(x+h*u)-function(x-h*u))/(2*h),v)
        rows.append(abs(left-right)/max(abs(left),abs(right),1e-14))
    return rows


class AdjointTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(782)
        self.a=self.rng.normal(size=(2,64))+1j*self.rng.normal(size=(2,64))
        self.u=self.rng.normal(size=(2,64))+1j*self.rng.normal(size=(2,64))
        self.v=self.rng.normal(size=(2,64))+1j*self.rng.normal(size=(2,64))
        self.c=config('CNT_OC',.2)

    def test_shell_with_exact_mock_tangent(self):
        n,cells=64,3
        s,ts=3.7,.81
        T=self.a/np.linalg.norm(self.a);Tt=self.u/np.linalg.norm(self.u)
        res=SimpleNamespace(n=n,cells=cells,scale=s,time_scale=ts,
                            omega=2*np.pi*np.fft.fftfreq(n,.2),template=T,time_tangent=Tt)
        def unpack(x): return s*(x[:2*n]+1j*x[2*n:4*n]).reshape(2,n),np.sqrt(cells)*x[4*n:-2],x[-2],ts*x[-1]
        res.unpack=unpack
        x=self.rng.normal(size=4*n+cells+2)
        u=self.rng.normal(size=len(x));v=self.rng.normal(size=len(x))
        # Arbitrary linear engine with cross field/population couplings.
        K=self.rng.normal(size=(4*n+cells,4*n+cells))*.03
        a,p,ph,t=unpack(x);ap=np.r_[a.real.ravel(),a.imag.ravel(),p]
        out=K@ap;b0=(out[:2*n]+1j*out[2*n:4*n]).reshape(2,n)
        ua,up,uph,ut=unpack(u);dout=K@np.r_[ua.real.ravel(),ua.imag.ravel(),up]
        db0=(dout[:2*n]+1j*dout[2*n:4*n]).reshape(2,n)
        factor=np.exp(1j*res.omega*t-1j*ph);b=np.fft.ifft(np.fft.fft(b0)*factor)
        db=np.fft.ifft(np.fft.fft(db0)*factor)+np.fft.ifft(np.fft.fft(b)*1j*res.omega)*ut-1j*b*uph
        da=(db-ua)/s
        tangent=np.r_[da.real.ravel(),da.imag.ravel(),(up-dout[4*n:])/np.sqrt(cells),inner(1j*T,ua/s),inner(Tt,ua/s)]
        vb,vn,g=shell_vjp(res,x,b0,v)
        back=K.T@np.r_[vb.real.ravel(),vb.imag.ravel(),vn]
        g[:4*n]+=s*back[:4*n];g[4*n:-2]+=np.sqrt(cells)*back[4*n:]
        self.assertLess(abs(tangent@v-u@g)/max(abs(tangent@v),abs(u@g)),1e-12)

    def test_linear(self):
        n=self.a.shape[-1]
        for value,back in [(np.fft.fft(self.u),n*np.fft.ifft(self.v)),
                           (np.fft.ifft(self.u),np.fft.fft(self.v)/n)]:
            self.assertLess(abs(inner(value,self.v)-inner(self.u,back))/abs(inner(self.u,back)),1e-12)
        h=np.exp(.2j*self.rng.normal(size=self.a.shape))*.9
        J=self.rng.normal(size=(2,2))+1j*self.rng.normal(size=(2,2))
        f=lambda a:J@np.fft.ifft(h*np.fft.fft(a))
        back=np.fft.ifft(h.conj()*np.fft.fft(J.conj().T@self.v))
        self.assertLess(abs(inner(f(self.u),self.v)-inner(self.u,back))/abs(inner(self.u,back)),1e-12)

    def test_kerr(self):
        np.testing.assert_allclose(kerr_value(self.a,.17),kerr(self.a[None],.17)[0],rtol=1e-14,atol=1e-14)
        self.assertLess(min(dot_scan(lambda a:kerr_value(a,.17),lambda a,v:kerr_vjp(a,v,.17),self.a,self.u,self.v)),1e-7)

    def test_cnt(self):
        value=cnt_value(self.a,self.c,.125)[0]
        expected=absorber(self.a[None],np.array([self.c['sa_modulation']]),self.c,.125)[0][0]
        np.testing.assert_allclose(value,expected,rtol=1e-14,atol=1e-14)
        self.assertLess(min(dot_scan(lambda a:cnt_value(a,self.c,.125)[0],lambda a,v:cnt_vjp(a,v,self.c,.125),self.a,self.u,self.v)),1e-7)

    def test_full_blocks(self):
        f=CavityResidual(self.c,.125,self.a,.0275,.8)
        x=f.pack(self.a,np.full(f.cells,.3),.15,.1)
        adj=DiscreteAdjoint(f)
        r,g=adj.value_and_vjp(x)
        np.testing.assert_allclose(r,f(x),rtol=1e-10,atol=1e-12)
        for kind in ['full','field_u','pop_u','phase_time_u','field_v','pop_v','gauge_v']:
            u=self.rng.normal(size=len(x));v=self.rng.normal(size=len(x))
            if kind=='field_u':u[4*f.n:]=0
            if kind=='pop_u':u[:4*f.n]=0;u[-2:]=0
            if kind=='phase_time_u':u[:-2]=0
            if kind=='field_v':v[4*f.n:]=0
            if kind=='pop_v':v[:4*f.n]=0;v[-2:]=0
            if kind=='gauge_v':v[:-2]=0
            u/=np.linalg.norm(u);v/=np.linalg.norm(v)
            right=u@f.vjp(x,v)
            errors=[]
            for h in SCALES:
                left=((f(x+h*u)-f(x-h*u))/(2*h))@v
                errors.append(abs(left-right)/max(abs(left),abs(right),1e-14))
            self.assertLess(min(errors),1e-5,msg=(kind,errors))

    def test_edf_population_and_optical_cotangents(self):
        f=CavityResidual(self.c,.125,self.a,.0275,.8)
        adj=DiscreteAdjoint(f);n=f.n
        initial=np.fft.fft(self.a);pop=np.full(f.cells,.3)
        x=np.r_[initial.real.ravel(),initial.imag.ravel(),pop]
        def unpack(x):return (x[:2*n]+1j*x[2*n:4*n]).reshape(2,n),x[4*n:]
        def forward(x):
            field,p=unpack(x);b,neq,_=adj.segment(field,p,1)
            return np.r_[b.real.ravel(),b.imag.ravel(),neq]
        def reverse(x,v):
            field,p=unpack(x);vb,vn=unpack(v)
            a,z=adj.segment_vjp(field,p,1,vb,vn)
            return np.r_[a.real.ravel(),a.imag.ravel(),z]
        for block in ['all','only_population_output','only_population_input']:
            u=self.rng.normal(size=len(x));v=self.rng.normal(size=len(x))
            if block=='only_population_output':v[:4*n]=0
            if block=='only_population_input':u[:4*n]=0
            u/=np.linalg.norm(u);v/=np.linalg.norm(v)
            self.assertLess(min(dot_scan(forward,reverse,x,u,v)),1e-5)

    def test_cnt_late_output_depends_on_early_input(self):
        c=dict(self.c,sa_recovery_ps=10.)
        a=self.a[:,:8];u=np.zeros_like(a);u[:,0]=self.u[:,0]
        v=np.zeros_like(a);v[:,-1]=self.v[:,-1]
        back=cnt_vjp(a,v,c,.125)
        self.assertGreater(np.linalg.norm(back[:,0]),1e-12)
        self.assertLess(min(dot_scan(lambda a:cnt_value(a,c,.125)[0],lambda a,v:cnt_vjp(a,v,c,.125),a,u,v)),1e-5)

    def test_oc_before_cnt(self):
        c=dict(self.c,topology='OC_CNT')
        f=CavityResidual(c,.125,self.a,.0275,.8)
        x=f.pack(self.a,np.full(f.cells,.3),.15,.1)
        r,_=DiscreteAdjoint(f).value_and_vjp(x)
        np.testing.assert_allclose(r,f(x),rtol=1e-10,atol=1e-12)
        u=self.rng.normal(size=len(x));u/=np.linalg.norm(u)
        v=self.rng.normal(size=len(x));v/=np.linalg.norm(v)
        self.assertLess(min(dot_scan(f,lambda x,v:f.vjp(x,v),x,u,v)),1e-5)


if __name__=='__main__':unittest.main()
