#!/usr/bin/env python3
"""Concurrency sweep + throughput probe for the v0.24.0 validation.

For each concurrency level: fire c simultaneous chat completions, report
aggregate + per-stream decode tok/s, verify the engine survives (/v1/models
still 200 and container running). Compare c=1 numbers against the v0.23.0
baseline (35B DFlash n=6 reference ~100-168 tok/s single-stream).
"""

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def one_request(model: str, max_tokens: int, prompt: str) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        }
    ).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    dt = time.time() - t0
    ct = out["usage"]["completion_tokens"]
    return {"secs": dt, "completion_tokens": ct, "tps": ct / dt}


def engine_alive(container: str) -> bool:
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as r:
            if r.status != 200:
                return False
    except Exception:
        return False
    p = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"name={container}"],
        capture_output=True,
        text=True,
    )
    return bool(p.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ornith")
    ap.add_argument("--container", default="ornith-v0240-test")
    ap.add_argument("--levels", default="1,4,8,12")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompt = "Write a detailed technical explanation of how a B-tree works."
    results = {}
    for c in [int(x) for x in args.levels.split(",")]:
        if not engine_alive(args.container):
            print(f"ENGINE DEAD before c={c}", flush=True)
            results[f"c{c}"] = "ENGINE-DEAD-BEFORE"
            break
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=c) as ex:
            futs = [
                ex.submit(one_request, args.model, args.max_tokens, prompt)
                for _ in range(c)
            ]
            done, errs = [], []
            for f in futs:
                try:
                    done.append(f.result())
                except Exception as e:  # noqa: BLE001
                    errs.append(str(e)[:200])
        wall = time.time() - t0
        total_tokens = sum(d["completion_tokens"] for d in done)
        agg = total_tokens / wall if wall else 0
        per = sorted(d["tps"] for d in done)
        med = per[len(per) // 2] if per else 0
        alive = engine_alive(args.container)
        results[f"c{c}"] = {
            "ok": len(done),
            "errors": errs,
            "wall_secs": round(wall, 1),
            "aggregate_tps": round(agg, 1),
            "median_stream_tps": round(med, 1),
            "engine_alive_after": alive,
        }
        print(
            f"c={c:3d}  ok={len(done):3d} err={len(errs)}  "
            f"agg={agg:7.1f} tok/s  med-stream={med:6.1f} tok/s  "
            f"alive={alive}",
            flush=True,
        )
        if not alive:
            print(f"ENGINE DIED at c={c}", flush=True)
            break
        time.sleep(3)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    dead = any(
        v == "ENGINE-DEAD-BEFORE" or (isinstance(v, dict) and not v["engine_alive_after"])
        for v in results.values()
    )
    return 1 if dead else 0


if __name__ == "__main__":
    sys.exit(main())
