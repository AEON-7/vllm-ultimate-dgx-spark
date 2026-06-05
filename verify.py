"""Self-contained verification of the aeon-vllm-ultimate image."""
import sys

print("=" * 60)
print("AEON vLLM Ultimate — environment verification")
print("=" * 60)

try:
    import vllm
    print(f"  vllm:        {vllm.__version__}  ({vllm.__file__})")
except Exception as e:
    print(f"  vllm:        FAIL ({e})")
    sys.exit(1)

try:
    import flashinfer
    print(f"  flashinfer:  {flashinfer.__version__}")
except Exception as e:
    print(f"  flashinfer:  WARN ({e})")

try:
    import torch
    print(f"  torch:       {torch.__version__}  (cuda {torch.version.cuda})")
    print(f"  cuda avail:  {torch.cuda.is_available()}")
except Exception as e:
    print(f"  torch:       FAIL ({e})")
    sys.exit(1)

try:
    import modelopt
    print(f"  modelopt:    {modelopt.__version__}")
except Exception as e:
    print(f"  modelopt:    WARN ({e})")

try:
    import turboquant
    ver = getattr(turboquant, "__version__", "unknown")
    print(f"  turboquant:  {ver}")
except ImportError as e:
    print(f"  turboquant:  not installed ({e})")

try:
    from vllm import LLM, SamplingParams
    from vllm.config import VllmConfig
    print(f"  vllm.LLM:    importable")
except Exception as e:
    print(f"  vllm.LLM:    FAIL ({e})")
    sys.exit(1)

print()
print("GREEN - aeon-vllm-ultimate ready")
