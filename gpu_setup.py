"""Optional standard CuPy installation; no machine-specific CUDA paths."""

try:
    import cupy as cp
except ImportError as exc:
    raise ImportError(
        "GPU mode requires CuPy matching your CUDA installation; use CPU mode otherwise."
    ) from exc
