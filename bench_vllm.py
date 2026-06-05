#!/usr/bin/env python3
"""Single-stream + concurrent benchmark for AEON vLLM Ultimate.

Measures:
  - TTFT  (time to first token, ms)
  - TPOT  (time per output token after the first, ms/tok)
  - Aggregate output tok/s (across the run)

Single-stream pass: one request at a time, N rounds.
Concurrent pass: K parallel requests fired together, repeated.

Usage:
    python3 bench_vllm.py \
        --base http://localhost:8000 \
        --model aeon \
        --label "MTP+NVFP4-KV" \
        --single-rounds 5 \
        --concurrent-streams 4 \
        --concurrent-rounds 3 \
        --max-tokens 256
"""
import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import aiohttp


PROMPTS = [
    "Explain how an internal combustion engine works in 8 short sentences.",
    "Write a 12-line poem about a lighthouse during a thunderstorm.",
    "List 10 practical productivity tips, each as one short sentence.",
    "Outline the plot of a heist movie set in Tokyo with 4 bullet points.",
    "Describe the structure of a tropical hurricane in 6 short paragraphs.",
    "Give a one-paragraph crash course on the difference between TCP and UDP.",
    "Write a short letter from a captain to her crew before a long voyage.",
    "Sketch a recipe for a vegetarian ramen, with ingredients then 7 steps.",
]


@dataclass
class StreamMetrics:
    ttft_ms: float = 0.0
    output_tokens: int = 0
    output_text_chars: int = 0
    wall_s: float = 0.0
    error: str = ""

    @property
    def tpot_ms(self) -> float:
        # time per output token AFTER the first
        if self.output_tokens <= 1:
            return 0.0
        decode_s = self.wall_s - (self.ttft_ms / 1000.0)
        return (decode_s / max(1, self.output_tokens - 1)) * 1000.0

    @property
    def toks_per_s(self) -> float:
        if self.wall_s <= 0:
            return 0.0
        return self.output_tokens / self.wall_s


async def stream_one(
    session: aiohttp.ClientSession,
    base: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> StreamMetrics:
    m = StreamMetrics()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }
    start = time.perf_counter()
    first_token_time = 0.0
    try:
        async with session.post(
            f"{base}/v1/chat/completions",
            json=body,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            if resp.status != 200:
                m.error = f"HTTP {resp.status}: {await resp.text()}"
                m.wall_s = time.perf_counter() - start
                return m
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except Exception:
                    continue
                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                content = delta.get("content") or ""
                if content:
                    if first_token_time == 0.0:
                        first_token_time = time.perf_counter() - start
                        m.ttft_ms = first_token_time * 1000.0
                    m.output_text_chars += len(content)
                # usage may come in the final chunk
                usage = chunk.get("usage")
                if usage:
                    m.output_tokens = int(usage.get("completion_tokens", m.output_tokens))
            m.wall_s = time.perf_counter() - start
    except Exception as e:
        m.error = repr(e)
        m.wall_s = time.perf_counter() - start
    # fallback token count from chars if usage didn't include it
    if m.output_tokens == 0 and m.output_text_chars > 0:
        m.output_tokens = max(1, m.output_text_chars // 4)
    return m


async def run_single(
    base: str, model: str, rounds: int, max_tokens: int
) -> list[StreamMetrics]:
    print(f"\n=== single-stream  rounds={rounds}  max_tokens={max_tokens} ===")
    results = []
    async with aiohttp.ClientSession() as session:
        # Warmup (not counted)
        await stream_one(session, base, model, PROMPTS[0], max_tokens=16)
        for i in range(rounds):
            prompt = PROMPTS[i % len(PROMPTS)]
            m = await stream_one(session, base, model, prompt, max_tokens)
            print(
                f"  [{i+1}/{rounds}] TTFT={m.ttft_ms:7.1f}ms  "
                f"TPOT={m.tpot_ms:6.2f}ms/tok  "
                f"tok/s={m.toks_per_s:6.2f}  out_toks={m.output_tokens}"
                + (f"  ERR={m.error}" if m.error else "")
            )
            results.append(m)
    return results


async def run_concurrent(
    base: str, model: str, streams: int, rounds: int, max_tokens: int
) -> list[StreamMetrics]:
    print(f"\n=== concurrent  streams={streams}  rounds={rounds}  max_tokens={max_tokens} ===")
    all_results = []
    async with aiohttp.ClientSession() as session:
        for i in range(rounds):
            prompts = [PROMPTS[(i * streams + j) % len(PROMPTS)] for j in range(streams)]
            t0 = time.perf_counter()
            tasks = [
                stream_one(session, base, model, p, max_tokens) for p in prompts
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            elapsed = time.perf_counter() - t0
            total_toks = sum(m.output_tokens for m in results)
            agg_toks_s = total_toks / elapsed if elapsed > 0 else 0.0
            avg_ttft = statistics.mean(m.ttft_ms for m in results if not m.error) if results else 0.0
            print(
                f"  [round {i+1}/{rounds}] wall={elapsed:6.2f}s  "
                f"agg_toks={total_toks}  agg_tok/s={agg_toks_s:7.2f}  "
                f"avg_TTFT={avg_ttft:7.1f}ms"
            )
            for j, m in enumerate(results):
                print(
                    f"     stream {j}: TTFT={m.ttft_ms:7.1f}ms "
                    f"TPOT={m.tpot_ms:6.2f}  tok/s={m.toks_per_s:6.2f}  "
                    f"out={m.output_tokens}"
                    + (f"  ERR={m.error}" if m.error else "")
                )
            all_results.extend(results)
    return all_results


def summarize(label: str, results: list[StreamMetrics], kind: str) -> dict:
    ok = [r for r in results if not r.error and r.output_tokens > 0]
    if not ok:
        return {"label": label, "kind": kind, "n": 0, "error": "no successful streams"}

    ttfts = [r.ttft_ms for r in ok]
    tpots = [r.tpot_ms for r in ok]
    tps = [r.toks_per_s for r in ok]
    total_toks = sum(r.output_tokens for r in ok)

    def st(xs):
        return {
            "mean": statistics.mean(xs),
            "median": statistics.median(xs),
            "min": min(xs),
            "max": max(xs),
            "p95": sorted(xs)[int(len(xs) * 0.95)] if len(xs) >= 2 else xs[0],
        }

    return {
        "label": label,
        "kind": kind,
        "n": len(ok),
        "total_output_tokens": total_toks,
        "TTFT_ms": st(ttfts),
        "TPOT_ms": st(tpots),
        "toks_per_s": st(tps),
    }


def print_summary(s: dict):
    print(f"\n--- summary: {s['kind']} {s['label']} ---")
    if "error" in s:
        print(f"  error: {s['error']}")
        return
    print(f"  n={s['n']}  total_out_toks={s['total_output_tokens']}")
    for key in ("TTFT_ms", "TPOT_ms", "toks_per_s"):
        v = s[key]
        print(
            f"  {key:12s}  mean={v['mean']:8.2f}  median={v['median']:8.2f}  "
            f"min={v['min']:8.2f}  max={v['max']:8.2f}  p95={v['p95']:8.2f}"
        )


async def main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="aeon")
    ap.add_argument("--label", default="aeon-vllm-ultimate")
    ap.add_argument("--single-rounds", type=int, default=5)
    ap.add_argument("--concurrent-streams", type=int, default=4)
    ap.add_argument("--concurrent-rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print(f"=== bench {args.label} → {args.base} model={args.model} ===")

    single = await run_single(
        args.base, args.model, args.single_rounds, args.max_tokens
    )
    concurrent = await run_concurrent(
        args.base, args.model, args.concurrent_streams, args.concurrent_rounds, args.max_tokens
    )

    summary = {
        "label": args.label,
        "single": summarize(args.label, single, "single"),
        "concurrent": summarize(args.label, concurrent, f"concurrent×{args.concurrent_streams}"),
    }
    print_summary(summary["single"])
    print_summary(summary["concurrent"])

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[saved] {args.out}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
