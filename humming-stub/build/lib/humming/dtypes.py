class DataType:
    """Stub. Calling this will fail — only used to satisfy `from humming.dtypes import DataType`."""
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "humming-stub: real humming library is not installed. "
            "Choose a quantization method other than `humming` (e.g. modelopt, awq, gguf)."
        )
