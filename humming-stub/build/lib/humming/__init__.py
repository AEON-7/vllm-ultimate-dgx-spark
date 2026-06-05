"""Stub for the humming quantization library.

The real humming kernels are NVIDIA-internal / research. vLLM imports symbol
names at module load time (gated only by `current_platform.is_cuda()`), so on
any CUDA build we need these names to exist or the vLLM quantization registry
fails to import — even if our chosen quant method isn't humming.

Calling any of these stub symbols will raise. We only need the imports to
succeed; actual humming usage isn't supported.
"""
__version__ = "0.0.0-aeon-stub"
