"""Three nonzero pulse guesses at g08_c53 parameters, on a fixed fine grid."""
import json, time, hashlib
import numpy as np
from scipy.signal import find_peaks
from config import ROOT, PROJECT, config, map_initial
from steady_state import CavityResidual, solve
from adaptive_solver import health


def run():
    c = config('CNT_OC', .2)
    n, dt, pump, oc = 8192, .125, .05, .8
    t = (np.arange(n)-n/2)*dt
    results = []
    for width, energy in [(10., 100.), (30., 100.), (30., 400.)]:
        # Initial cavity energy pJ, not prescribed output or renormalized evolution.
        a, pop, _ = map_initial(c, [oc], n=n, dt=dt)
        shape = np.exp(-t*t/(4*width**2))
        a[0,0] = shape*np.sqrt(energy/(dt*np.sum(shape**2)))
        residual = CavityResidual(c, dt, a[0], pump, oc, gpu=True)
        start = time.perf_counter()
        x, history, status = solve(residual, residual.pack(a[0],pop[0]), max_steps=20)
        r = residual(x)
        b, p, phase, shift = residual.unpack(x)
        power = np.sum(abs(residual.output)**2, axis=0)
        peaks = find_peaks(power, prominence=.1*max(power.max(),1e-100))[0]
        boundary = health(b[None], dt)
        row = dict(width_ps=width, initial_cavity_energy_pJ=energy, status=status,
                   seconds=time.perf_counter()-start, evaluations=residual.evaluations,
                   residual=float(np.linalg.norm(r)),
                   relative_field_residual=float(np.linalg.norm(r[:4*n])*residual.scale/max(np.linalg.norm(b),1e-100)),
                   population_gap=float(np.max(abs(p-residual.neq))),
                   output_energy_nJ=float(power.sum()*dt/1000), peaks=int(len(peaks)),
                   time_edge=boundary[0], spectral_edge=boundary[1],
                   phase_rad=float(phase), shift_ps=float(shift), history=history,
                   certified_stable=False)
        results.append(row)
        np.savez_compressed(ROOT/f'steady_trial_{len(results)}.npz', a=b,pop=p,x=x,dt=dt)
        payload = dict(topology='CNT_OC',oc=oc,pump_W=pump,gdd_ps2=.2,n=n,dt_ps=dt,
                       source_sha256={f:hashlib.sha256((PROJECT/f).read_bytes()).hexdigest()
                                      for f in ['steady_state.py','spectral_engine.py','run_steady_pilot.py']},
                       trials=results)
        (ROOT/'steady_pilot.json').write_text(json.dumps(payload,indent=2),encoding='utf8')
        print(json.dumps({k:v for k,v in row.items() if k!='history'}), flush=True)


if __name__ == '__main__':
    run()
