"""Finite-bank exhaustive comparisons and rank-aware novelty diagnostics."""
import numpy as np


def exhaustive_history(model, count, radius):
    """Compare equal cardinalities; unrestricted best is the full nested space."""
    rows=[]
    for mask in range(1 << count):
        ids=[i for i in range(count) if mask & (1 << i)]
        rows.append(dict(selected=ids,count=len(ids),prediction=float(model.solve(ids,radius)[1])))
    best=[max((r for r in rows if r['count']==k),key=lambda r:r['prediction']) for k in range(count+1)]
    return dict(subsets=rows,best_by_count=best)


def novelty(basis, vector, tolerance=1e-12):
    """SVD projection, excluding residual-target columns and numerical null modes."""
    u,s,vh=np.linalg.svd(basis,full_matrices=False)
    keep=s>s[0]*tolerance
    coefficients=vh[keep].T@((u[:,keep].T@vector)/s[keep])
    error=vector-basis@coefficients
    return dict(relative=float(np.linalg.norm(error)/np.linalg.norm(vector)),rank=int(np.sum(keep)),
                relative_singular=(s/s[0]).tolist()),coefficients


def paired_novelty(states,responses,direction,response,tolerance=1e-12):
    """Independent projections plus a shared-coefficient diagnostic with stated units.

    Joint blocks are divided by the corresponding target-vector norms. This
    is an explanatory fit, not a new trust-region metric or physical equation.
    """
    nx,cx=novelty(states,direction,tolerance)
    nr,cr=novelty(responses,response,tolerance)
    stacked=np.vstack([states/np.linalg.norm(direction),responses/np.linalg.norm(response)])
    target=np.r_[direction/np.linalg.norm(direction),response/np.linalg.norm(response)]
    joint,c=novelty(stacked,target,tolerance)
    return dict(state=nx,response=nr,joint=joint,
        response_error_from_state_coefficients=float(np.linalg.norm(responses@cx-response)/np.linalg.norm(response)),
        state_error_from_response_coefficients=float(np.linalg.norm(states@cr-direction)/np.linalg.norm(direction)),
        joint_state_error=float(np.linalg.norm(states@c-direction)/np.linalg.norm(direction)),
        joint_response_error=float(np.linalg.norm(responses@c-response)/np.linalg.norm(response)))
