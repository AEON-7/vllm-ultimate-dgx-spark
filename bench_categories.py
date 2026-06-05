#!/usr/bin/env python3
"""Categorized benchmark for AEON vLLM Ultimate.

Measures TTFT / TPOT / tok/s broken down by prompt category
(reasoning, math, code, prose, dialogue, summary). Helps catch cases
where a generic bench masks per-domain speculator acceptance differences.

Usage:
    python3 bench_categories.py \\
        --base http://localhost:8000 \\
        --model aeon-test \\
        --label "MTP-XS body + DFlash n=15 greedy" \\
        --temperature 0 \\
        --max-tokens 400 \\
        --concurrent-streams 4 \\
        --concurrent-rounds 2 \\
        --out bench_categories.json
"""
import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import aiohttp


PROMPTS_BY_CATEGORY = {
    "reasoning": [
        "A train leaves city A at 60 mph heading east. Another train leaves city B at 80 mph heading west on the same track. The cities are 350 miles apart. When and where do they meet? Show your reasoning step by step.",
        "If all squares are rectangles, and some rectangles are red, can we logically conclude that some squares are red? Walk through your reasoning carefully.",
        "Five people sit in a row. Alice is not next to Bob. Carol is next to Alice. Dan is at one end. Eve is to the right of Bob. Where does each person sit? Explain.",
        "You have two ropes, each takes exactly 60 minutes to burn end-to-end but burns unevenly. Using only a lighter, measure exactly 45 minutes. Explain.",
    ],
    "math": [
        "Compute the derivative of f(x) = 3x^4 - 5x^2 + 7x - 2 using the power rule. Show each step.",
        "Evaluate the integral of (2x + 3) dx from x=1 to x=4. Show the antiderivative and the substitution.",
        "Prove that the sum of the infinite series 1/2 + 1/4 + 1/8 + 1/16 + ... equals 1, using the geometric series formula.",
        "Find all real solutions to x^3 - 6x^2 + 11x - 6 = 0. Factor and show roots.",
    ],
    "code": [
        "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number using memoization. Include a brief docstring and one example call.",
        "Implement quicksort in JavaScript. Include comments explaining the partition step and the recursion.",
        "Write a Rust function that takes a &str and returns the count of each ASCII letter as a HashMap<char, u32>. Handle case-insensitively.",
        "Write a SQL query against tables `orders(id, customer_id, total, created_at)` and `customers(id, name)` that returns the top 5 customers by lifetime total spend, with their names. Use a CTE.",
    ],
    "prose": [
        "Describe the formation and structure of a tropical hurricane in 6 short paragraphs, suitable for an introductory atmospheric science course.",
        "Write a 12-line poem about a lighthouse during a thunderstorm. Use vivid imagery and a consistent meter.",
        "Write the opening 3 paragraphs of a short story set in a small coastal town where the tide hasn't come in for three days.",
        "Write a letter from a sailing captain to her crew on the eve of a long voyage. About 200 words. Warm but resolute tone.",
    ],
    "dialogue": [
        "Write a 5-turn dialogue between an old librarian and a curious child about why people still tell stories. Each turn should be 1-2 sentences.",
        "Two scientists argue about whether time travel could ever be possible. Write 6 turns. One is enthusiastic, the other skeptical.",
        "A customer-service phone call: a customer is frustrated about a delayed package; the agent must remain calm and helpful. Write 4 turns each.",
        "A teacher explains the concept of recursion to a confused student using an analogy. Write a 6-turn exchange where the student gradually gets it.",
    ],
    "summary": [
        "Summarize the Big Bang theory and the main lines of observational evidence supporting it in exactly 5 bullet points.",
        "Give a 200-word overview of the difference between supervised and unsupervised learning in machine learning, with one example of each.",
        "Summarize the plot of Hamlet in 3 paragraphs. Keep it neutral and clear, no spoilers omitted.",
        "Summarize the key arguments for and against universal basic income in a single page of structured bullet points (for / against / open questions).",
    ],
}


@dataclass
class StreamMetrics:
    ttft_ms: float = 0.0
    output_tokens: int = 0
    output_text_chars: int = 0
    wall_s: float = 0.0
    error: str = ""
    category: str = ""

    @property
    def tpot_ms(self) -> float:
        if self.output_tokens <= 1:
            return 0.0
        decode_s = self.wall_s - (self.ttft_ms / 1000.0)
        return (decode_s / max(1, self.output_tokens - 1)) * 1000.0

    @property
    def decode_tok_s(self) -> float:
        """Decode-only token rate (excludes TTFT). 1000/TPOT."""
        if self.tpot_ms <= 0:
            return 0.0
        return 1000.0 / self.tpot_ms

    @property
    def wall_tok_s(self) -> float:
        """End-to-end token rate (includes TTFT)."""
        if self.wall_s <= 0:
            return 0.0
        return self.output_tokens / self.wall_s


async def stream_one(
    session: aiohttp.ClientSession,
    base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    category: str = "",
) -> StreamMetrics:
    m = StreamMetrics(category=category)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": temperature,
    }
    start = time.perf_counter()
    first_token_time = 0.0
    try:
        async with session.post(
            f"{base}/v1/chat/completions",
            json=body,
            timeout=aiohttp.ClientTimeout(total=300),
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
                usage = chunk.get("usage")
                if usage:
                    m.output_tokens = int(usage.get("completion_tokens", m.output_tokens))
            m.wall_s = time.perf_counter() - start
    except Exception as e:
        m.error = repr(e)
        m.wall_s = time.perf_counter() - start
    if m.output_tokens == 0 and m.output_text_chars > 0:
        m.output_tokens = max(1, m.output_text_chars // 4)
    return m


async def run_single_per_category(
    base: str, model: str, max_tokens: int, temperature: float
) -> dict[str, list[StreamMetrics]]:
    print(f"\n=== single-stream by category  max_tokens={max_tokens}  temperature={temperature} ===")
    out: dict[str, list[StreamMetrics]] = {}
    async with aiohttp.ClientSession() as session:
        # global warmup
        await stream_one(session, base, model, "Hello.", 16, temperature, category="warmup")
        for cat, prompts in PROMPTS_BY_CATEGORY.items():
            print(f"\n  -- {cat} ({len(prompts)} prompts) --")
            out[cat] = []
            for i, p in enumerate(prompts):
                m = await stream_one(session, base, model, p, max_tokens, temperature, category=cat)
                print(
                    f"     [{i+1}/{len(prompts)}] TTFT={m.ttft_ms:7.1f}ms  "
                    f"TPOT={m.tpot_ms:6.2f}ms  decode={m.decode_tok_s:6.2f} tok/s  "
                    f"wall={m.wall_tok_s:6.2f} tok/s  out={m.output_tokens}"
                    + (f"  ERR={m.error}" if m.error else "")
                )
                out[cat].append(m)
    return out


async def run_concurrent_per_category(
    base: str, model: str, streams: int, rounds: int, max_tokens: int, temperature: float
) -> list[dict]:
    """Each round picks one prompt from each of `streams` categories and runs them concurrently."""
    print(f"\n=== concurrent  streams={streams}  rounds={rounds}  max_tokens={max_tokens}  temperature={temperature} ===")
    cat_names = list(PROMPTS_BY_CATEGORY.keys())
    rounds_out = []
    async with aiohttp.ClientSession() as session:
        for r in range(rounds):
            picks = []
            for j in range(streams):
                cat = cat_names[(r * streams + j) % len(cat_names)]
                prompts = PROMPTS_BY_CATEGORY[cat]
                p = prompts[(r) % len(prompts)]
                picks.append((cat, p))
            t0 = time.perf_counter()
            tasks = [
                stream_one(session, base, model, p, max_tokens, temperature, category=cat)
                for (cat, p) in picks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            elapsed = time.perf_counter() - t0
            total_toks = sum(m.output_tokens for m in results)
            agg_toks_s = total_toks / elapsed if elapsed > 0 else 0.0
            avg_ttft = statistics.mean(m.ttft_ms for m in results if not m.error) if results else 0.0
            print(
                f"  [round {r+1}/{rounds}] wall={elapsed:6.2f}s  "
                f"agg_toks={total_toks}  agg_tok/s={agg_toks_s:7.2f}  "
                f"avg_TTFT={avg_ttft:7.1f}ms"
            )
            for (cat, _), m in zip(picks, results):
                print(
                    f"     {cat:10s}: TTFT={m.ttft_ms:7.1f}ms TPOT={m.tpot_ms:6.2f}  "
                    f"decode={m.decode_tok_s:6.2f}  wall={m.wall_tok_s:6.2f}  out={m.output_tokens}"
                    + (f"  ERR={m.error}" if m.error else "")
                )
            rounds_out.append({
                "round": r + 1,
                "elapsed_s": elapsed,
                "agg_tok_s": agg_toks_s,
                "avg_ttft_ms": avg_ttft,
                "streams": [
                    {"category": cat, "ttft_ms": m.ttft_ms, "tpot_ms": m.tpot_ms,
                     "decode_tok_s": m.decode_tok_s, "wall_tok_s": m.wall_tok_s,
                     "output_tokens": m.output_tokens, "error": m.error}
                    for (cat, _), m in zip(picks, results)
                ],
            })
    return rounds_out


def stat(xs: list[float]) -> dict:
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "min": min(xs),
        "max": max(xs),
    }


def summarize_single(single: dict[str, list[StreamMetrics]]) -> dict:
    by_cat = {}
    all_decode, all_wall, all_ttft, all_tpot = [], [], [], []
    for cat, results in single.items():
        ok = [r for r in results if not r.error and r.output_tokens > 1]
        by_cat[cat] = {
            "n": len(ok),
            "ttft_ms": stat([r.ttft_ms for r in ok]),
            "tpot_ms": stat([r.tpot_ms for r in ok]),
            "decode_tok_s": stat([r.decode_tok_s for r in ok]),
            "wall_tok_s": stat([r.wall_tok_s for r in ok]),
            "output_tokens": stat([r.output_tokens for r in ok]),
        }
        all_decode.extend([r.decode_tok_s for r in ok])
        all_wall.extend([r.wall_tok_s for r in ok])
        all_ttft.extend([r.ttft_ms for r in ok])
        all_tpot.extend([r.tpot_ms for r in ok])
    return {
        "by_category": by_cat,
        "overall": {
            "ttft_ms": stat(all_ttft),
            "tpot_ms": stat(all_tpot),
            "decode_tok_s": stat(all_decode),
            "wall_tok_s": stat(all_wall),
        },
    }


def print_single_table(s: dict):
    print("\n--- summary: single-stream BY CATEGORY ---")
    hdr = f"  {'category':10s}  {'n':>3s}  {'TTFTmed':>9s}  {'TPOTmed':>9s}  {'decode μ':>9s}  {'decode med':>11s}  {'wall μ':>8s}  {'out_toks μ':>11s}"
    print(hdr)
    for cat, v in s["by_category"].items():
        if v["n"] == 0:
            continue
        print(
            f"  {cat:10s}  {v['n']:>3d}  "
            f"{v['ttft_ms']['median']:>9.1f}  {v['tpot_ms']['median']:>9.2f}  "
            f"{v['decode_tok_s']['mean']:>9.2f}  {v['decode_tok_s']['median']:>11.2f}  "
            f"{v['wall_tok_s']['mean']:>8.2f}  {v['output_tokens']['mean']:>11.1f}"
        )
    o = s["overall"]
    if o["decode_tok_s"].get("n"):
        print(
            f"  {'OVERALL':10s}  {o['decode_tok_s']['n']:>3d}  "
            f"{o['ttft_ms']['median']:>9.1f}  {o['tpot_ms']['median']:>9.2f}  "
            f"{o['decode_tok_s']['mean']:>9.2f}  {o['decode_tok_s']['median']:>11.2f}  "
            f"{o['wall_tok_s']['mean']:>8.2f}  {'-':>11s}"
        )


async def main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="aeon-test")
    ap.add_argument("--label", default="aeon-vllm-ultimate")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--concurrent-streams", type=int, default=4)
    ap.add_argument("--concurrent-rounds", type=int, default=2)
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-single", action="store_true")
    ap.add_argument("--skip-concurrent", action="store_true")
    args = ap.parse_args()

    print(f"=== bench {args.label} → {args.base} model={args.model} temp={args.temperature} ===")

    single_summary, concurrent_rounds = {}, []
    single = {}
    if not args.skip_single:
        single = await run_single_per_category(args.base, args.model, args.max_tokens, args.temperature)
        single_summary = summarize_single(single)
        print_single_table(single_summary)

    if not args.skip_concurrent:
        concurrent_rounds = await run_concurrent_per_category(
            args.base, args.model, args.concurrent_streams, args.concurrent_rounds,
            args.max_tokens, args.temperature
        )

    payload = {
        "label": args.label,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "single": single_summary,
        "concurrent": concurrent_rounds,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[saved] {args.out}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
