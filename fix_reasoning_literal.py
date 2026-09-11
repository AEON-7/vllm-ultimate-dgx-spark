from pathlib import Path
import re
p = Path("/usr/local/lib/python3.12/site-packages/vllm/sampling_params.py")
t = p.read_text()
if not re.search(r"from typing(?:_extensions)? import[^\n]*\bLiteral\b", t):
    if "from __future__ import annotations" in t:
        t = t.replace(
            "from __future__ import annotations\n",
            "from __future__ import annotations\nfrom typing import Literal\n",
            1,
        )
    else:
        t = "from typing import Literal\n" + t
    p.write_text(t)
    print("patched Literal import in site-packages")
else:
    print("Literal import already present in site-packages")
from vllm.sampling_params import SamplingParams, validate_reasoning_eos_policy
assert validate_reasoning_eos_policy(None) == "force_end"
assert SamplingParams().reasoning_eos_policy == "force_end"
print("reasoning_eos smoke OK")
