"""Second window check at the NEW endpoint; no assumption from input padding."""
import json
import numpy as np
from config import ROOT,config
from steady_window import restore,embed,center

state=np.load(ROOT/'tail_continuation.npz')
original,x=restore(config('CNT_OC',.2),state,gpu=True)
wide,xw=restore(config('CNT_OC',.2),state,wide=True,gpu=True)
r=original(x);rw=wide(xw);n=original.n;norm=np.linalg.norm(r)
result=dict(original_window_ps=n*.125,wide_window_ps=2*n*.125,
    original_residual=float(norm),wide_residual=float(np.linalg.norm(rw)),
    relative_norm_change=float(abs(np.linalg.norm(rw)/norm-1)),
    center_vector_relative_change=float(np.linalg.norm(center(rw,n)-r)/norm),
    full_vector_relative_change=float(np.linalg.norm(rw-embed(r,n))/norm),
    outer_residual_norm=float(np.linalg.norm(rw-embed(center(rw,n),n))))
(ROOT/'final_window_check.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result),flush=True)
