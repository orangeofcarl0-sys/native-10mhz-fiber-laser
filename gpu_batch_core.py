"""Vectorized independent cavities; batch axis never mixes physical fields."""

from gpu_setup import cp as np
from cupyx.scipy.fft import fft, ifft
from python_engine import Engine

_kerr_kernel = np.RawKernel(
    r"""
extern "C" __global__ void kerr_step(const double2* a,double2* out,int n,int count,double g){
    int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=count)return;
    int b=i/n,j=i%n,ix=b*2*n+j,iy=ix+n;
    double2 x=a[ix],y=a[iy];double s=0.7071067811865475244;
    double pr=(x.x-y.y)*s,pi=(x.y+y.x)*s;
    double mr=(x.x+y.y)*s,mi=(x.y-y.x)*s;
    double pp=pr*pr+pi*pi,mm=mr*mr+mi*mi;
    double ap=(2.0/3.0)*g*(pp+2*mm),am=(2.0/3.0)*g*(mm+2*pp);
    double sp,cp,sm,cm;sincos(ap,&sp,&cp);sincos(am,&sm,&cm);
    double rp=pr*cp-pi*sp,ip=pr*sp+pi*cp;
    double rm=mr*cm-mi*sm,im=mr*sm+mi*cm;
    out[ix]=make_double2((rp+rm)*s,(ip+im)*s);
    out[iy]=make_double2((ip-im)*s,(rm-rp)*s);
}
""",
    "kerr_step",
)


def kerr(a, g):
    """Same circular-polarization Kerr rotation as CPU, fused without changing physics."""
    import numpy as hostnp

    a = np.ascontiguousarray(a)
    out = np.empty_like(a)
    n = a.shape[-1]
    count = a.shape[0] * n
    _kerr_kernel(
        ((count + 255) // 256,),
        (256,),
        (a, out, hostnp.int32(n), hostnp.int32(count), hostnp.float64(g)),
    )
    return out


_cnt_kernel = np.RawKernel(
    r"""
extern "C" __global__ void cnt_step(const double* p,double* q,double* transmission,
    double* minimum,int batch,int n,double tau,double es,double modulation,double nonsat,double dt){
    int b=blockDim.x*blockIdx.x+threadIdx.x;if(b>=batch)return;
    double state=q[b],low=modulation;
    for(int j=0;j<n;j++){
        int index=b*n+j;double rate=1.0/tau+p[index]/es;
        double equilibrium=modulation/(tau*rate),decay=exp(-rate*dt);
        double middle=equilibrium+(state-equilibrium)*sqrt(decay);
        transmission[index]=1.0-nonsat-middle;low=fmin(low,middle);
        state=equilibrium+(state-equilibrium)*decay;
    }
    q[b]=state;minimum[b]=low;
}
""",
    "cnt_step",
)


def absorber(a, q, c, dt):
    import numpy as hostnp

    power = np.sum(abs(a) ** 2, axis=1)
    batch, n = power.shape
    transmission = np.empty_like(power)
    minimum = np.empty(batch, dtype=np.float64)
    q = q.copy()
    _cnt_kernel(
        ((batch + 31) // 32,),
        (32,),
        (
            power,
            q,
            transmission,
            minimum,
            hostnp.int32(batch),
            hostnp.int32(n),
            hostnp.float64(c["sa_recovery_ps"]),
            hostnp.float64(c["sa_saturation_energy_pJ"]),
            hostnp.float64(c["sa_modulation"]),
            hostnp.float64(c["sa_nonsaturable"]),
            hostnp.float64(dt),
        ),
    )
    # Invalid transmission becomes NaN and is checked on device by the resident runner.
    metrics = np.stack([power.sum(axis=1) * dt, power.max(axis=1), minimum], axis=1)
    return (
        a * np.sqrt(np.where(transmission > 0, transmission, np.nan)[:, None, :]),
        q,
        metrics,
    )


class BatchEngine:
    def __init__(self, c, dt, n, pumps, oc):
        self.c = c
        self.dt = dt
        self.n = n
        self.e = Engine(c, dt, n)
        self.e.profile = np.asarray(self.e.profile)
        self.e.w = np.asarray(self.e.w)
        self.e.J = np.asarray(self.e.J)
        self.e.ops = [
            (steps, dz, np.asarray(half), np.asarray(R), gamma)
            for steps, dz, half, R, gamma in self.e.ops
        ]
        self.pumps = np.asarray(pumps)
        self.oc = np.asarray(oc)
        self.last_gap = np.zeros(len(oc))
        self.last_tau = np.zeros(len(oc))

    def fiber(self, a, index, pump, pop):
        e = self.e
        c = self.c
        steps, dz, half, R, gamma = e.ops[index]
        if index != 1:
            full = half**2
            a = ifft(half * fft(R.T @ a, axis=-1), axis=-1)
            for j in range(steps):
                a = kerr(a, gamma * dz)
                a = ifft((half if j == steps - 1 else full) * fft(a, axis=-1), axis=-1)
            return R @ a, pump, pop
        gaps = []
        taus = []
        for j in range(steps):
            inv = pop[:, j].copy()
            spectrum_in = fft(R.T @ a, axis=-1)
            old = (
                np.sum(np.sum(abs(spectrum_in) ** 2, axis=1) * e.profile, axis=1)
                * self.dt
                * 1e-12
                * e.rep
                / e.n
            )
            gain = (c["emission_s_m"] * inv - c["alpha_s_m"] * (1 - inv))[
                :, None
            ] * e.profile
            hh = half[None, :, :] * np.exp(gain[:, None, :] * dz / 4)
            a = R @ ifft(hh * spectrum_in, axis=-1)
            a = kerr(a, gamma * dz)
            spectrum_out = hh * fft(R.T @ a, axis=-1)
            a = R @ ifft(spectrum_out, axis=-1)
            pmid = pump * np.exp(-c["alpha_p_m"] * (1 - inv) * dz / 2)
            new = (
                np.sum(np.sum(abs(spectrum_out) ** 2, axis=1) * e.profile, axis=1)
                * self.dt
                * 1e-12
                * e.rep
                / e.n
            )
            smid = np.sqrt(old * new)
            A = (c["alpha_p_m"] * pmid / e.hp + c["alpha_s_m"] * smid / e.hs) / e.ions
            B = (
                1 / c["upper_lifetime_s"]
                + (
                    c["alpha_p_m"] * pmid / e.hp
                    + (c["alpha_s_m"] + c["emission_s_m"]) * smid / e.hs
                )
                / e.ions
            )
            equilibrium = A / B
            if c.get("gain_mode", "dynamic") != "frozen":
                pop[:, j] = inv + (equilibrium - inv) * (-np.expm1(-B / e.rep))
            pump *= np.exp(-c["alpha_p_m"] * (1 - inv) * dz)
            gaps.append(abs(inv - equilibrium))
            taus.append(1 / B)
        self.last_gap = np.max(np.stack(gaps), axis=0)
        self.last_tau = np.max(np.stack(taus), axis=0)
        return a, pump, pop

    def split(self, a):
        a = a * 10 ** (-self.c["oc_excess_dB"] / 20)
        return (
            a * np.sqrt(1 - self.oc)[:, None, None],
            a * np.sqrt(self.oc)[:, None, None],
        )

    def step(self, a, pop, q):
        c = self.c
        e = self.e
        pump = self.pumps * 10 ** (-c["pump_path_dB"] / 10)
        for j in range(6):
            a, pump, pop = self.fiber(a, j, pump, pop)
            if j == 3:
                a = e.J @ a
                if c["topology"] == "OC_CNT":
                    a, out = self.split(a)
                a, q, sa = absorber(a, q, c, self.dt)
                if c["topology"] == "CNT_OC":
                    a, out = self.split(a)
                a *= 10 ** (-c["splice_connector_dB"] / 20)
        a *= 10 ** (-c["hybrid_IL_dB"] / 20)
        a[:, 1] *= 10 ** (-c["hybrid_PDL_dB"] / 20)
        q = c["sa_modulation"] + (q - c["sa_modulation"]) * np.exp(
            -max(0, 1e12 / e.rep - self.n * self.dt) / c["sa_recovery_ps"]
        )
        return a, out, pop, q, sa


def aligned_residual(previous, current):
    p = np.sum(abs(previous) ** 2, axis=1)
    q = np.sum(abs(current) ** 2, axis=1)
    n = p.shape[1]
    row = np.arange(len(p))
    corr = ifft(fft(p, axis=1) * np.conj(fft(q, axis=1)), axis=1).real
    k = np.argmax(corr, axis=1)
    den = corr[row, (k - 1) % n] - 2 * corr[row, k] + corr[row, (k + 1) % n]
    fraction = np.divide(
        0.5 * (corr[row, (k - 1) % n] - corr[row, (k + 1) % n]),
        den,
        out=np.zeros(len(p)),
        where=den != 0,
    )
    shift = np.where(k < n / 2, k, k - n) + fraction
    shifted = ifft(
        fft(current, axis=-1)
        * np.exp(-2j * np.pi * np.fft.fftfreq(n)[None, None, :] * shift[:, None, None]),
        axis=-1,
    )
    error = np.linalg.norm(np.sum(abs(shifted) ** 2, axis=1) - p, axis=1) / np.maximum(
        np.linalg.norm(p, axis=1), 1e-100
    )
    return error, shift
