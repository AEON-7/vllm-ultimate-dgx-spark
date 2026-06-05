# vLLM source pin

Build was against:

- **Repo**: `lesj0610/vllm`
- **Branch**: `lesj/triton-nvfp4-kv-fork-20260602`
- **Commit**: `e8c77b85`
- **Upstream PR**: [vllm-project/vllm#44389](https://github.com/vllm-project/vllm/pull/44389) — Triton software NVFP4 KV cache (~3× capacity)

To reproduce the build:

```bash
git clone https://github.com/lesj0610/vllm.git vllm-src
cd vllm-src
git checkout lesj/triton-nvfp4-kv-fork-20260602
git checkout e8c77b85
cd ..
# Then docker build -t aeon-vllm-ultimate:latest .
```

The full source is not vendored in this repo (~140 MB) — only the patches, Dockerfile, humming-stub, verify script, bench tooling, and bench artifacts.
