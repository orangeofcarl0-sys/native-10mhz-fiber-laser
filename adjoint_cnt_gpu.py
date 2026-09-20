"""Serial scalar recurrence on GPU; optical cotangents stay vectorized in CuPy."""
from gpu_setup import cp

forward = cp.RawKernel(r'''
extern "C" __global__ void cnt_forward(const double* eq, const double* h,
 double* before, double* mid, int n, double q) {
 for(int j=0;j<n;++j) {
  before[j]=q; mid[j]=eq[j]+(q-eq[j])*h[j];
  q=eq[j]+(q-eq[j])*h[j]*h[j];
 }
}''', 'cnt_forward')

backward = cp.RawKernel(r'''
extern "C" __global__ void cnt_backward(const double* rate,const double* eq,
 const double* h,const double* before,const double* midbar,double* powerbar,
 int n,double dt,double es) {
 double qb=0.;
 for(int j=n-1;j>=0;--j) {
  double hj=h[j], vm=midbar[j];
  double eb=vm*(1-hj)+qb*(1-hj*hj);
  double hb=vm*(before[j]-eq[j])+2*hj*qb*(before[j]-eq[j]);
  powerbar[j]=(-eb*eq[j]/rate[j]-hb*dt*hj/2)/es;
  qb=vm*hj+qb*hj*hj;
 }
}''', 'cnt_backward')
