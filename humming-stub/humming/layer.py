class HummingLayerMeta(type):
    """Stub metaclass."""
    pass

class HummingMethod:
    """Stub for humming layer method."""
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "humming-stub: real humming library is not installed. "
            "Choose a different quantization method (e.g. modelopt, awq, gguf)."
        )
