# Overlay build. The base tag is the local 2026-11-13 v0.33.0-omni build on the AEON GB10 nodes (not published separately).
FROM aeon-vllm-ultimate:2026-11-13-v0.33.0-omni
# AEON 2026-11-23: DFlash2 speculative decoding for Qwen3.8-Flash-Next (qwen4_exp)
#  - qwen4_exp/nvidia/model.py: EAGLE3-style aux hidden-state hook (HC-stream mean, HC-SP aware)
#  - model_executor/models/config.py: allow method "dflash" for Qwen4Exp
COPY model.py /opt/venv/lib/python3.12/site-packages/vllm/models/qwen4_exp/nvidia/model.py
COPY config.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/config.py
COPY aeon-dflash2-qwen4exp.patch /opt/aeon/patches/aeon-dflash2-qwen4exp.patch
LABEL org.opencontainers.image.source="https://github.com/AEON-7/vllm-ultimate-dgx-spark" \
      org.opencontainers.image.version="2026-11-23-v0.33.0-omni-dflash2" \
      org.opencontainers.image.description="AEON vLLM Ultimate v0.33.0 (DGX Spark sm_121a) + DFlash2 for Qwen3.8-Flash-Next"
