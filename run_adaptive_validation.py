"""Bounded physical pilots; this is not a rescan of the 640-point map."""

from pathlib import Path
import numpy as np
from attractor_search import run_point

if __name__ == "__main__":
    f = np.load(Path(__file__).parent / "tests/fixtures/near_candidate.npz")
    state = (f["a"][None], f["pop"][None], np.array([0.05]))
    run_point(
        state=state,
        dt=float(f["dt"]),
        n=f["a"].shape[-1],
        search_blocks=0,
        direct_rounds=80,
        name="adaptive_pilot_checkpoint",
    )
    run_point(
        pump=0.001,
        search_blocks=1,
        block_rounds=8,
        direct_rounds=64,
        name="adaptive_pilot_lowpump",
    )
