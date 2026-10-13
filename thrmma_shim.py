# --- compat shim (appended): re-export classes relocated in cutlass-dsl 4.6.0 ---
# 4.5.x exposed ThrMma/TiledMma/ThrCopy/TiledCopy under cutlass.cute.core; 4.6.0 moved
# them to the cutlass.cute top level. Lazy module __getattr__ (PEP 562) avoids any
# circular-import hazard during package init.
def __getattr__(name):
    if name in ("ThrMma", "TiledMma", "ThrCopy", "TiledCopy"):
        import cutlass.cute as _cute
        return getattr(_cute, name)
    raise AttributeError(f"module 'cutlass.cute.core' has no attribute {name!r}")
