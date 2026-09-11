"""Humming dtypes stub for Spark images without real humming-kernels.

vLLM 0.29 humming_utils builds dtype maps at import time when `humming` is
importable. We expose the same module-level tokens so ModelOpt / MXFP4 import
paths succeed; constructing DataType() for real compute still fails loudly.
"""


class _StubDtype:
    __slots__ = ("name",)

    def __init__(self, name: str):
        object.__setattr__(self, "name", name)

    def __repr__(self) -> str:
        return f"<humming-stub.dtypes.{self.name}>"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StubDtype) and self.name == other.name


class DataType:
    """Stub class. Instantiating for real work fails ? use a different quant method."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "humming-stub: real humming library is not installed. "
            "Choose a quantization method other than `humming` "
            "(e.g. modelopt, awq, gguf)."
        )


# Tokens referenced by vllm.model_executor.layers.quantization.utils.humming_utils
float4e2m1 = _StubDtype("float4e2m1")
float8e4m3 = _StubDtype("float8e4m3")
float8e5m2 = _StubDtype("float8e5m2")
float8e8m0 = _StubDtype("float8e8m0")
float16 = _StubDtype("float16")
bfloat16 = _StubDtype("bfloat16")
float32 = _StubDtype("float32")
int8 = _StubDtype("int8")
uint2 = _StubDtype("uint2")
uint3 = _StubDtype("uint3")
uint4 = _StubDtype("uint4")
uint5 = _StubDtype("uint5")
uint6 = _StubDtype("uint6")
uint7 = _StubDtype("uint7")
uint8 = _StubDtype("uint8")
