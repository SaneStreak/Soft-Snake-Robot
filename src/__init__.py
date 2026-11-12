"""Hierarchical CPG-Regulated RL for a Continuum Soft Snake Robot."""

# Initialize PyTorch OpenMP runtime before conda MKL-linked numpy/scipy on Windows.
try:
    import torch as _torch  # noqa: F401
except ImportError:
    pass
