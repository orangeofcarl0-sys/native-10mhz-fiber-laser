"""CPU/GPU residual equivalence and finite-difference directional consistency."""
import json
import numpy as np
from config import ROOT, config, map_initial
from steady_state import CavityResidual

rows = []
for topology in ('OC_CNT', 'CNT_OC'):
    c = config(topology, .2)
    a, p, q = map_initial(c, [.8], n=512, dt=.5)
    cpu = CavityResidual(c, .5, a[0], .05, .8)
    gpu = CavityResidual(c, .5, a[0], .05, .8, gpu=True)
    x = cpu.pack(a[0], p[0], .1, .05)
    rc, rg = cpu(x), gpu(x)
    error = np.linalg.norm(rc-rg)/np.linalg.norm(rc)
    np.testing.assert_allclose(rc, rg, atol=1e-11, rtol=1e-9)
    v = np.random.default_rng(42).normal(size=x.size)
    v /= np.linalg.norm(v)
    # Includes field, population and both smooth gauge coordinates.
    reference = (gpu(x+1e-5*v)-gpu(x-1e-5*v))/2e-5
    forward = (gpu(x+1e-7*v)-rg)/1e-7
    discrepancy = np.linalg.norm(forward-reference)/np.linalg.norm(reference)
    assert discrepancy < 1e-4
    rows.append(dict(topology=topology, cpu_gpu_relative=float(error),
                     jv_relative_discrepancy=float(discrepancy)))
(ROOT/'steady_validation.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
print(json.dumps(rows),flush=True)
