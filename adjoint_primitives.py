"""Real-inner-product pullbacks; fields have shape (2, N), units sqrt(W)."""
import numpy as np


def shell_value(res, x, b0, neq):
    a, pop, phase, shift = res.unpack(x)
    factor = np.exp(1j*res.omega*shift-1j*phase)
    b = np.fft.ifft(np.fft.fft(b0)*factor)
    r = ((b-a)/res.scale).ravel()
    delta = a/res.scale-res.template
    gauges = [np.vdot(t, delta).real for t in [1j*res.template, res.time_tangent]]
    return np.r_[r.real, r.imag, (pop-neq)/np.sqrt(res.cells), gauges]


def shell_vjp(res, x, b0, v):
    """Return engine output cotangents and direct packed-state contribution."""
    n, s = res.n, res.scale
    _, _, phase, shift = res.unpack(x)
    factor = np.exp(1j*res.omega*shift-1j*phase)
    spectrum = np.fft.fft(b0)*factor
    b = np.fft.ifft(spectrum)
    vb = (v[:2*n]+1j*v[2*n:4*n]).reshape(2, n)/s
    vb0 = np.fft.ifft(np.fft.fft(vb)*factor.conj())
    vn = v[4*n:-2]
    direct = -s*vb + v[-2]*1j*res.template + v[-1]*res.time_tangent
    phase_bar = np.vdot(vb, -1j*b).real
    time_bar = res.time_scale*np.vdot(vb, np.fft.ifft(1j*res.omega*spectrum)).real
    return vb0, -vn/np.sqrt(res.cells), np.r_[
        direct.real.ravel(), direct.imag.ravel(), vn, phase_bar, time_bar]


def kerr_value(a, g, xp=np):
    p = (a[0]+1j*a[1])/np.sqrt(2)
    m = (a[0]-1j*a[1])/np.sqrt(2)
    pp, mm = xp.abs(p)**2, xp.abs(m)**2
    p = p*xp.exp(2j*g/3*(pp+2*mm))
    m = m*xp.exp(2j*g/3*(mm+2*pp))
    return xp.stack([(p+m)/np.sqrt(2), (p-m)/(1j*np.sqrt(2))])


def kerr_vjp(a, v, g, xp=np):
    p = (a[0]+1j*a[1])/np.sqrt(2)
    m = (a[0]-1j*a[1])/np.sqrt(2)
    vp = (v[0]+1j*v[1])/np.sqrt(2)
    vm = (v[0]-1j*v[1])/np.sqrt(2)
    alpha = 2*g/3
    ep = xp.exp(1j*alpha*(xp.abs(p)**2+2*xp.abs(m)**2))
    em = xp.exp(1j*alpha*(xp.abs(m)**2+2*xp.abs(p)**2))
    sp = xp.real(xp.conj(vp)*1j*p*ep)
    sm = xp.real(xp.conj(vm)*1j*m*em)
    vp = ep.conj()*vp + (2*alpha*sp+4*alpha*sm)*p
    vm = em.conj()*vm + (4*alpha*sp+2*alpha*sm)*m
    return xp.stack([(vp+vm)/np.sqrt(2), (vp-vm)/(1j*np.sqrt(2))])


def cnt_value(a, c, dt, xp=np):
    """Physical recurrence with fixed q_initial and no terminal-q objective."""
    power = xp.sum(xp.abs(a)**2, axis=0)
    rate = 1/c['sa_recovery_ps'] + power/c['sa_saturation_energy_pJ']
    eq = c['sa_modulation']/(c['sa_recovery_ps']*rate)
    h = xp.exp(-rate*dt/2)
    before, midpoint = xp.empty_like(power), xp.empty_like(power)
    if xp is np:
        q = c['sa_modulation']
        for j in range(len(power)):
            before[j] = q
            midpoint[j] = eq[j]+(q-eq[j])*h[j]
            q = eq[j]+(q-eq[j])*h[j]**2
    else:
        from adjoint_cnt_gpu import forward
        forward((1,), (1,), (eq,h,before,midpoint,np.int32(len(power)),np.float64(c['sa_modulation'])))
    transmission = xp.sqrt(1-c['sa_nonsaturable']-midpoint)
    return a*transmission, (rate,eq,h,before,transmission)


def cnt_vjp(a, v, c, dt, xp=np):
    _, (rate,eq,h,before,t) = cnt_value(a,c,dt,xp)
    midpoint_bar = -xp.sum(xp.real(xp.conj(v)*a),axis=0)/(2*t)
    power_bar = xp.empty_like(t)
    if xp is np:
        qbar = 0.  # Terminal q and monitoring metrics are dead outputs.
        for j in range(len(t)-1,-1,-1):
            vm, hj = midpoint_bar[j], h[j]
            eqbar = vm*(1-hj)+qbar*(1-hj**2)
            hbar = vm*(before[j]-eq[j])+2*hj*qbar*(before[j]-eq[j])
            power_bar[j] = (-eqbar*eq[j]/rate[j]-hbar*dt*hj/2)/c['sa_saturation_energy_pJ']
            qbar = vm*hj+qbar*hj**2
    else:
        from adjoint_cnt_gpu import backward
        backward((1,), (1,), (rate,eq,h,before,midpoint_bar,power_bar,
                 np.int32(len(t)),np.float64(dt),np.float64(c['sa_saturation_energy_pJ'])))
    return v*t+2*a*power_bar
