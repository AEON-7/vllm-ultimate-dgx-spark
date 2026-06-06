#!/usr/bin/env python3
"""Capability eval through a running vLLM OpenAI endpoint.

Reuses the dataset loaders + HumanEval exec sandbox + IFEval constraints from
capability_eval.py, but swaps the model-call layer to HTTP so we can eval the
NVFP4 model (which doesn't load in plain transformers) through the exact vLLM
serving path users hit.

MMLU is scored by argmax over the A/B/C/D first-token logprobs returned by
vLLM's chat/completions (logprobs + top_logprobs), matching the original
first-token-logprob methodology.

Usage:
    python3 capability_eval_vllm.py \
        --base http://localhost:8000 --model gemma12b --label NVFP4 \
        --mmlu-n 100 --he-n 40 --if-n 50 --out out.json
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

CONCURRENCY = 12  # parallel API requests (vLLM serves 16 concurrent)

# Reuse the dataset loaders + scoring harness from the transformers eval
sys.path.insert(0, "/work")
from capability_eval import (  # noqa: E402
    load_mmlu,
    load_humaneval,
    run_humaneval_problem,
    IFEVAL_TASKS,
)
from datasets import load_dataset  # noqa: E402


def load_mmlu_balanced(per_subject: int) -> list[dict]:
    """Balanced MMLU across ALL 57 subjects (per_subject questions each).

    The streaming 'all' loader returns abstract_algebra-first (a pathological
    worst case for quantized models). This loads the full test set and samples
    evenly across every subject for a fair, diverse measurement.
    """
    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subj: dict[str, list] = {}
    for ex in ds:
        by_subj.setdefault(ex["subject"], []).append(ex)
    out = []
    for subj in sorted(by_subj):
        for ex in by_subj[subj][:per_subject]:
            out.append({
                "question": ex["question"],
                "choices": ex["choices"],
                "answer": "ABCD"[ex["answer"]],
                "subject": ex["subject"],
            })
    return out


def api_chat(base: str, model: str, prompt: str, max_tokens: int = 512,
             logprobs: bool = False, top_logprobs: int = 0) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = top_logprobs
    r = requests.post(f"{base}/v1/chat/completions", json=body, timeout=300)
    r.raise_for_status()
    return r.json()


def api_generate(base: str, model: str, prompt: str, max_tokens: int = 512) -> str:
    j = api_chat(base, model, prompt, max_tokens=max_tokens)
    return (j["choices"][0]["message"]["content"] or "").strip()


# ---- MMLU (argmax over A/B/C/D first-token logprobs) ----

def _mmlu_one(base, model, q):
    prompt = (
        "Answer this multiple choice question with just A, B, C, or D.\n\n"
        f"{q['question']}\n\n"
        + "\n".join(f"{l}. {c}" for l, c in zip("ABCD", q["choices"]))
        + "\n\nAnswer:"
    )
    j = api_chat(base, model, prompt, max_tokens=1, logprobs=True, top_logprobs=20)
    pick = None
    try:
        content_lp = j["choices"][0]["logprobs"]["content"]
        if content_lp:
            cand = {}
            for tl in content_lp[0]["top_logprobs"]:
                tok = tl["token"].strip().upper()
                if tok in ("A", "B", "C", "D") and tok not in cand:
                    cand[tok] = tl["logprob"]
            if cand:
                pick = max(cand.items(), key=lambda kv: kv[1])[0]
    except (KeyError, IndexError, TypeError):
        pass
    if pick is None:
        txt = (j["choices"][0]["message"]["content"] or "").strip().upper()
        m = re.search(r"[ABCD]", txt)
        pick = m.group(0) if m else "?"
    return pick == q["answer"]


def run_mmlu(base, model, questions):
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        oks = list(ex.map(lambda q: _mmlu_one(base, model, q), questions))
    correct = sum(oks)
    by_subject = {}
    for q, ok in zip(questions, oks):
        s = by_subject.setdefault(q["subject"], {"total": 0, "correct": 0})
        s["total"] += 1
        s["correct"] += int(ok)
    return {"accuracy": correct / len(questions), "correct": correct,
            "total": len(questions), "by_subject": by_subject}


# ---- HumanEval (reuse exec sandbox) ----

def _he_gen(base, model, p):
    return api_generate(
        base, model,
        f"Complete this Python function. Write only the function definition:\n\n```python\n{p['prompt']}```",
        max_tokens=512,
    )


def run_humaneval(base, model, problems):
    # Parallelize generation (the slow part); exec scoring is fast + sequential
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        gens = list(ex.map(lambda p: _he_gen(base, model, p), problems))
    results, syn, fun = [], 0, 0
    for p, gen in zip(problems, gens):
        r = run_humaneval_problem(p["prompt"], gen, p["test"], p["entry_point"])
        r["task_id"] = p["task_id"]
        results.append(r)
        syn += int(r["syntactic"])
        fun += int(r["functional"])
    return {"syntactic_pass": syn / len(problems), "functional_pass": fun / len(problems),
            "syntactic_count": syn, "functional_count": fun, "total": len(problems),
            "sample_results": results[:5]}


# ---- IFEval (reuse constraints) ----

def run_ifeval(base, model, n):
    tasks = (IFEVAL_TASKS * ((n // len(IFEVAL_TASKS)) + 1))[:n]
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        gens = list(ex.map(lambda t: api_generate(base, model, t["prompt"], max_tokens=200), tasks))
    passed, fails = 0, []
    for t, gen in zip(tasks, gens):
        try:
            ok = bool(t["check"](gen))
        except Exception:
            ok = False
        if ok:
            passed += 1
        elif len(fails) < 5:
            fails.append({"prompt": t["prompt"][:80], "gen": gen[:200]})
    return {"pass_rate": passed / len(tasks), "passed": passed, "total": len(tasks),
            "sample_failures": fails}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="gemma12b")
    ap.add_argument("--label", default="model")
    ap.add_argument("--mmlu-n", type=int, default=100)
    ap.add_argument("--mmlu-per-subject", type=int, default=0,
                    help="If >0, use a balanced MMLU of N per subject across all 57 subjects")
    ap.add_argument("--he-n", type=int, default=40)
    ap.add_argument("--if-n", type=int, default=50)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print(f"=== capability eval (vLLM API) — {args.label} ===", flush=True)
    print("loading datasets...", flush=True)
    if args.mmlu_per_subject > 0:
        mmlu = load_mmlu_balanced(args.mmlu_per_subject)
        print(f"  MMLU balanced: {args.mmlu_per_subject}/subject across "
              f"{len(set(q['subject'] for q in mmlu))} subjects = {len(mmlu)} Q", flush=True)
    else:
        mmlu = load_mmlu(args.mmlu_n)
    he = load_humaneval(args.he_n)
    print(f"  MMLU={len(mmlu)}  HumanEval={len(he)}  IFEval={args.if_n}", flush=True)

    t0 = time.time()
    print(f"\n[MMLU] {len(mmlu)} questions...", flush=True)
    mmlu_r = run_mmlu(args.base, args.model, mmlu)
    print(f"  accuracy = {mmlu_r['accuracy']:.3f} ({mmlu_r['correct']}/{mmlu_r['total']})  [{time.time()-t0:.0f}s]", flush=True)

    t0 = time.time()
    print(f"\n[HumanEval] {len(he)} problems...", flush=True)
    he_r = run_humaneval(args.base, args.model, he)
    print(f"  syntactic = {he_r['syntactic_pass']:.3f} ({he_r['syntactic_count']}/{he_r['total']})", flush=True)
    print(f"  functional = {he_r['functional_pass']:.3f} ({he_r['functional_count']}/{he_r['total']})  [{time.time()-t0:.0f}s]", flush=True)

    t0 = time.time()
    print(f"\n[IFEval] {args.if_n} tasks...", flush=True)
    if_r = run_ifeval(args.base, args.model, args.if_n)
    print(f"  pass_rate = {if_r['pass_rate']:.3f} ({if_r['passed']}/{if_r['total']})  [{time.time()-t0:.0f}s]", flush=True)

    summary = {"label": args.label, "model": args.model,
               "mmlu": mmlu_r, "humaneval": he_r, "ifeval": if_r}
    print("\n=== SUMMARY ===")
    print(f"  {args.label}: MMLU {mmlu_r['accuracy']*100:.1f}%  "
          f"HE-syn {he_r['syntactic_pass']*100:.1f}%  HE-fun {he_r['functional_pass']*100:.1f}%  "
          f"IFEval {if_r['pass_rate']*100:.1f}%")
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
