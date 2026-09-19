"""FP64 fused CNT: one block per cavity, affine block-prefix recurrence."""

import numpy as host
from gpu_setup import cp

_cnt_prefix = cp.RawKernel(
    r"""
extern "C" __global__ void cnt_prefix(const double2* field, double2* out,
 const double* qin, double* qout, double* metrics, int n,
 double tau, double es, double modulation, double nonsat, double dt) {
 int t=threadIdx.x, b=blockIdx.x, chunk=(n+255)/256;
 int start=t*chunk, end=min(n,start+chunk), base=2*b*n;
 __shared__ double sa[256],sb[256],sc[256];
 double A=1.,B=0.;
 for(int j=start;j<end;j++) {
  double2 x=field[base+j],y=field[base+n+j];
  double p=x.x*x.x+x.y*x.y+y.x*y.x+y.y*y.y;
  double rate=1./tau+p/es, eq=modulation/(tau*rate);
  double h=exp(-.5*rate*dt), a=h*h;
  A=a*A; B=a*B+eq*(1.-a);
 }
 sa[t]=A;sb[t]=B;__syncthreads();
 // Ordered inclusive scan of T_right o T_left; barriers prevent overwrite races.
 for(int offset=1;offset<256;offset*=2){
  double leftA=1.,leftB=0.;
  if(t>=offset){leftA=sa[t-offset];leftB=sb[t-offset];}
  double nextA=sa[t]*leftA,nextB=sa[t]*leftB+sb[t];
  __syncthreads();sa[t]=nextA;sb[t]=nextB;__syncthreads();
 }
 double state=t==0?qin[b]:sa[t-1]*qin[b]+sb[t-1];
 if(t==255)qout[b]=sa[t]*qin[b]+sb[t];
 __syncthreads();
 double energy=0.,peak=0.,low=modulation;
 for(int j=start;j<end;j++){
  double2 x=field[base+j],y=field[base+n+j];
  double p=x.x*x.x+x.y*x.y+y.x*y.x+y.y*y.y;
  double rate=1./tau+p/es,eq=modulation/(tau*rate);
  double h=exp(-.5*rate*dt),a=h*h;
  double middle=eq+(state-eq)*h,T=1.-nonsat-middle;
  double scale=T>0.?sqrt(T):nan("");
  out[base+j]=make_double2(x.x*scale,x.y*scale);
  out[base+n+j]=make_double2(y.x*scale,y.y*scale);
  state=eq+(state-eq)*a;
  energy+=p*dt;peak=fmax(peak,p);low=fmin(low,middle);
 }
 sa[t]=energy;sb[t]=peak;sc[t]=low;__syncthreads();
 for(int offset=128;offset>0;offset/=2){
  if(t<offset){sa[t]+=sa[t+offset];sb[t]=fmax(sb[t],sb[t+offset]);sc[t]=fmin(sc[t],sc[t+offset]);}
  __syncthreads();
 }
 if(t==0){metrics[3*b]=sa[0];metrics[3*b+1]=sb[0];metrics[3*b+2]=sc[0];}
}
""",
    "cnt_prefix",
)


def absorber_prefix(a, q, c, dt):
    if (
        a.dtype != cp.complex128
        or not a.flags.c_contiguous
        or a.ndim != 3
        or a.shape[1] != 2
        or q.dtype != cp.float64
        or q.shape != (len(a),)
        or not q.flags.c_contiguous
    ):
        raise ValueError("CNT prefix requires contiguous complex128 fields")
    out = cp.empty_like(a)
    qout = cp.empty_like(q)
    metrics = cp.empty((len(q), 3), dtype=cp.float64)
    _cnt_prefix(
        (len(q),),
        (256,),
        (
            a,
            out,
            q,
            qout,
            metrics,
            host.int32(a.shape[-1]),
            host.float64(c["sa_recovery_ps"]),
            host.float64(c["sa_saturation_energy_pJ"]),
            host.float64(c["sa_modulation"]),
            host.float64(c["sa_nonsaturable"]),
            host.float64(dt),
        ),
    )
    return out, qout, metrics


_weighted = cp.RawKernel(
    r"""
extern "C" __global__ void weighted(const double2* f,const double* profile,
 double* out,int n,double scale){
 int b=blockIdx.x,t=threadIdx.x;double sum=0.;
 __shared__ double sums[256];
 for(int j=t;j<n;j+=256){
  double2 x=f[2*b*n+j],y=f[2*b*n+n+j];
  sum+=(x.x*x.x+x.y*x.y+y.x*y.x+y.y*y.y)*profile[j];
 }
 sums[t]=sum;__syncthreads();
 for(int d=128;d>0;d/=2){if(t<d)sums[t]+=sums[t+d];__syncthreads();}
 if(t==0)out[b]=sums[0]*scale;
}
""",
    "weighted",
)

_gain_half = cp.RawKernel(
    r"""
extern "C" __global__ void gain_half(const double2* f,double2* out,
 const double2* half,const double* profile,const double* pop,
 int n,int cells,int cell,double emission,double absorption,double dz){
 int j=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y;if(j>=n)return;
 double inv=pop[b*cells+cell];
 double gain=exp((emission*inv-absorption*(1.-inv))*profile[j]*dz/4.);
 for(int p=0;p<2;p++){
  int i=2*b*n+p*n+j;double2 x=f[i],h=half[p*n+j];
  out[i]=make_double2(gain*(x.x*h.x-x.y*h.y),gain*(x.x*h.y+x.y*h.x));
 }
}
""",
    "gain_half",
)

_rate_update = cp.RawKernel(
    r"""
extern "C" __global__ void rate_update(const double* old,const double* next,
 double* pump,double* pop,double* gap,double* tau,int batch,int cells,int cell,
 double ap,double as,double em,double hp,double hs,double ions,double life,
 double rep,double dz,int frozen){
 int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=batch)return;
 double inv=pop[b*cells+cell],pmid=pump[b]*exp(-ap*(1.-inv)*dz/2.);
 double signal=sqrt(old[b]*next[b]);
 double A=(ap*pmid/hp+as*signal/hs)/ions;
 double B=1./life+(ap*pmid/hp+(as+em)*signal/hs)/ions;
 if(!frozen)pop[b*cells+cell]=inv+(A/B-inv)*(-expm1(-B/rep));
 pump[b]*=exp(-ap*(1.-inv)*dz);
 gap[b]=fmax(gap[b],fabs(inv-A/B));tau[b]=fmax(tau[b],1./B);
}
""",
    "rate_update",
)


def weighted_power(f, profile, scale):
    out = cp.empty(f.shape[0], dtype=cp.float64)
    _weighted(
        (f.shape[0],),
        (256,),
        (f, profile, out, host.int32(f.shape[-1]), host.float64(scale)),
    )
    return out


def gain_half(f, half, profile, pop, cell, c, dz):
    out = cp.empty_like(f)
    n = f.shape[-1]
    _gain_half(
        ((n + 255) // 256, f.shape[0]),
        (256,),
        (
            f,
            out,
            half,
            profile,
            pop,
            host.int32(n),
            host.int32(pop.shape[1]),
            host.int32(cell),
            host.float64(c["emission_s_m"]),
            host.float64(c["alpha_s_m"]),
            host.float64(dz),
        ),
    )
    return out


def rate_update(old, new, pump, pop, gap, tau, cell, c, e, dz):
    _rate_update(
        ((len(pump) + 255) // 256,),
        (256,),
        (
            old,
            new,
            pump,
            pop,
            gap,
            tau,
            host.int32(len(pump)),
            host.int32(pop.shape[1]),
            host.int32(cell),
            *[
                host.float64(x)
                for x in (
                    c["alpha_p_m"],
                    c["alpha_s_m"],
                    c["emission_s_m"],
                    e.hp,
                    e.hs,
                    e.ions,
                    c["upper_lifetime_s"],
                    e.rep,
                    dz,
                )
            ],
            host.int32(c.get("gain_mode", "dynamic") == "frozen"),
        ),
    )


_kerr_inplace = cp.RawKernel(
    r"""
extern "C" __global__ void kerr_inplace(double2* a,int n,double g){
 int j=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y;if(j>=n)return;
 int ix=2*b*n+j,iy=ix+n;double2 x=a[ix],y=a[iy];
 double s=0.7071067811865475244;
 double pr=(x.x-y.y)*s,pi=(x.y+y.x)*s,mr=(x.x+y.y)*s,mi=(x.y-y.x)*s;
 double pp=pr*pr+pi*pi,mm=mr*mr+mi*mi;
 double sp,cp_,sm,cm;sincos((2./3.)*g*(pp+2*mm),&sp,&cp_);
 sincos((2./3.)*g*(mm+2*pp),&sm,&cm);
 double rp=pr*cp_-pi*sp,ip=pr*sp+pi*cp_,rm=mr*cm-mi*sm,im=mr*sm+mi*cm;
 a[ix]=make_double2((rp+rm)*s,(ip+im)*s);
 a[iy]=make_double2((ip-im)*s,(rm-rp)*s);
}
""",
    "kerr_inplace",
)


def kerr_inplace(a, g):
    if a.dtype != cp.complex128 or not a.flags.c_contiguous:
        raise ValueError("In-place Kerr requires contiguous complex128 fields")
    _kerr_inplace(
        ((a.shape[-1] + 255) // 256, a.shape[0]),
        (256,),
        (a, host.int32(a.shape[-1]), host.float64(g)),
    )
    return a
