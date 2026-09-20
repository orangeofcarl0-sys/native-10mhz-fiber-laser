"""Exact discrete pullback of the frozen-population map, with segment replay.

No global tape: keep six segment inputs; replay only the segment being reversed.
All complex derivatives use the real inner product. No solver policy lives here.
"""
import numpy as np
from adjoint_primitives import shell_value, shell_vjp, kerr_value, kerr_vjp, cnt_value, cnt_vjp


class DiscreteAdjoint:
    def __init__(self, residual):
        self.res = residual
        self.e = residual.engine
        self.xp = self.e.xp
        self.c = self.e.c
        self.fft, self.ifft = self.e.fft, self.e.ifft
        self.n = residual.n
        self.weight = self.e.dt*1e-12*self.e.e.rep/self.n
        self.oc = np.sqrt(1-residual.oc)*10**(-self.c['oc_excess_dB']/20)
        self.loss = 10**(-self.c['hybrid_IL_dB']/20)*self.xp.asarray(
            [[1.], [10**(-self.c['hybrid_PDL_dB']/20)]])

    def power(self, f):
        return self.xp.sum(self.xp.abs(f)**2*self.e.profile)*self.weight

    def power_vjp(self, f, v):
        return 2*self.weight*self.e.profile*f*v

    def segment(self, f, pop, index, tape=False):
        """Primal segment; optional local tape is discarded after its reverse."""
        xp,c,e = self.xp,self.c,self.e
        steps,dz,half,r,gamma = e.ops[index]
        f = r.T@f
        local = []
        if index != 1:
            f = half*f
            for j in range(steps):
                a = self.ifft(f)
                if tape: local.append(a)
                f = (half if j==steps-1 else half*half)*self.fft(kerr_value(a,gamma*dz,xp))
            return r@f, None, local
        old = self.power(f)
        pump = xp.asarray(self.res.pump*10**(-c['pump_path_dB']/10))
        neq = []
        for j in range(steps):
            inv = pop[j]
            hh = half*xp.exp(((c['emission_s_m']+c['alpha_s_m'])*inv-c['alpha_s_m'])*e.profile*dz/4)
            a = r@self.ifft(hh*f)
            k = kerr_value(a,gamma*dz,xp)
            z = self.fft(r.T@k)
            newf = hh*z
            new = self.power(newf)
            signal = xp.sqrt(old*new)
            attenuation = xp.exp(-c['alpha_p_m']*(1-inv)*dz/2)
            pmid = pump*attenuation
            A = (c['alpha_p_m']*pmid/e.e.hp+c['alpha_s_m']*signal/e.e.hs)/e.e.ions
            B = 1/c['upper_lifetime_s']+(c['alpha_p_m']*pmid/e.e.hp+(c['alpha_s_m']+c['emission_s_m'])*signal/e.e.hs)/e.e.ions
            neq.append(A/B)
            if tape: local.append((f,hh,a,z,newf,old,new,signal,pump,attenuation,pmid,A,B))
            pump = pump*attenuation**2
            f,old = newf,new
        return r@f, xp.stack(neq), local

    def segment_vjp(self, f0, pop, index, vf, vn=None):
        xp,c,e = self.xp,self.c,self.e
        steps,dz,half,r,gamma = e.ops[index]
        _,_,tape = self.segment(f0,pop,index,tape=True)
        vf = r.conj().T@vf
        if index != 1:
            for j in range(steps-1,-1,-1):
                multiplier = half if j==steps-1 else half*half
                va = self.n*self.ifft(multiplier.conj()*vf)
                va = kerr_vjp(tape[j],va,gamma*dz,xp)
                vf = self.fft(va)/self.n
            return r@ (half.conj()*vf), xp.zeros_like(pop)
        vp = xp.zeros_like(pop)
        pump_bar = xp.asarray(0.)
        old_bar = xp.asarray(0.)
        for j in range(steps-1,-1,-1):
            f,hh,a,z,newf,old,new,signal,pump,attenuation,pmid,A,B = tape[j]
            abar, bbar = vn[j]/B, -vn[j]*A/B**2
            pmid_bar = (abar+bbar)*c['alpha_p_m']/e.e.hp/e.e.ions
            signal_bar = (abar*c['alpha_s_m']+bbar*(c['alpha_s_m']+c['emission_s_m']))/e.e.hs/e.e.ions
            # The archived nonzero-field states are differentiable. The zero
            # signal boundary uses the continuous zero optical gradient limit.
            denom = xp.maximum(signal, xp.finfo(xp.float64).tiny)
            new_bar = old_bar + signal_bar*old/(2*denom)
            old_bar = signal_bar*new/(2*denom)
            vp[j] += pump_bar*pump*attenuation**2*c['alpha_p_m']*dz
            vp[j] += pmid_bar*pmid*c['alpha_p_m']*dz/2
            pump_bar = pump_bar*attenuation**2+pmid_bar*attenuation
            vf = vf+self.power_vjp(newf,new_bar)
            dh = (c['emission_s_m']+c['alpha_s_m'])*e.profile*dz/4
            vp[j] += xp.sum(xp.real(xp.conj(vf)*newf)*dh)
            vk = r@(self.n*self.ifft(hh.conj()*vf))
            va = kerr_vjp(a,vk,gamma*dz,xp)
            vfirst = self.fft(r.conj().T@va)/self.n
            vp[j] += xp.sum(xp.real(xp.conj(vfirst)*(hh*f))*dh)
            vf = hh.conj()*vfirst
        vf += self.power_vjp(r.T@f0,old_bar)
        return r@vf,vp

    def junction(self, a):
        if self.c['topology']=='OC_CNT': a = a*self.oc
        a,_ = cnt_value(a,self.c,self.e.dt,self.xp)
        if self.c['topology']=='CNT_OC': a = a*self.oc
        return a

    def junction_vjp(self, a, v):
        if self.c['topology']=='CNT_OC': v = v*self.oc
        initial = a*self.oc if self.c['topology']=='OC_CNT' else a
        v = cnt_vjp(initial,v,self.c,self.e.dt,self.xp)
        return v*self.oc if self.c['topology']=='OC_CNT' else v

    def primal(self, a, pop):
        f = self.fft(a)
        checkpoints = []
        for j in range(6):
            checkpoints.append(f)
            f,nj,_ = self.segment(f,pop,j)
            if j==1: neq=nj
            if j==3:
                junction = self.e.J@self.ifft(f)
                f = self.fft(self.junction(junction))*10**(-self.c['splice_connector_dB']/20)
        return self.ifft(f*self.loss), neq, (checkpoints,junction)

    def reverse(self, pop, checkpoints, vb, vn):
        starts,junction = checkpoints
        vf = self.fft(vb)/self.n*self.loss
        vp = self.xp.zeros_like(pop)
        for j in range(5,-1,-1):
            if j==3:
                va = self.n*self.ifft(vf)*10**(-self.c['splice_connector_dB']/20)
                va = self.e.J.conj().T@self.junction_vjp(junction,va)
                vf = self.fft(va)/self.n
            vf,pj = self.segment_vjp(starts[j],pop,j,vf,vn if j==1 else None)
            vp += pj
        return self.n*self.ifft(vf),vp

    def value_and_vjp(self, x, v=None):
        """General cotangent; v=None is the first application J^T R(x)."""
        res,xp = self.res,self.xp
        a,pop,_,_ = res.unpack(x)
        pop = xp.asarray(pop)
        b,neq,checkpoints = self.primal(xp.asarray(a),pop)
        host = np.asarray if xp is np else xp.asnumpy
        b0 = host(b)
        value = shell_value(res,x,b0,host(neq))
        if v is None: v=value
        v=np.asarray(v)
        if v.shape!=x.shape: raise ValueError('Cotangent must match packed residual')
        vb,vn,g = shell_vjp(res,x,b0,v)
        va,vp = self.reverse(pop,checkpoints,xp.asarray(vb),xp.asarray(vn))
        va=host(va)*res.scale
        g[:2*res.n] += va.real.ravel()
        g[2*res.n:4*res.n] += va.imag.ravel()
        g[4*res.n:-2] += np.sqrt(res.cells)*host(vp)
        return value,g
