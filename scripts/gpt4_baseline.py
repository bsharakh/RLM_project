#!/usr/bin/env python
"""GPT-4 baseline: send full context + question in one call. Measures accuracy, time, cost."""
import argparse, os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from tests.musique_loader import load_musique_dataset

import re
_WORD = re.compile(r"[A-Za-z0-9]+")
def _toks(s): return {w.lower() for w in _WORD.findall(s or "")}

def _score(answer, golden):
    if not golden: return 0.0
    g = _toks(golden)
    if not g: return 0.0
    return len(_toks(answer) & g) / len(g)

# Pricing per 1M tokens (GPT-4o, as of 2025)
# Adjust these if using a different model
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4": {"input": 30.00, "output": 60.00},
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()

    ds = load_musique_dataset(split="validation", answerable_only=True)
    if args.hops:
        ds = [ex for ex in ds if ex.get("num_hops") == args.hops]
        print(f"Filtered to {len(ds)} {args.hops}-hop questions")

    subset = ds[args.idx : args.idx + args.num]
    print(f"Running {len(subset)} questions with {args.model} (full context baseline)\n")

    client = OpenAI()
    prices = PRICING.get(args.model, PRICING["gpt-4o"])

    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_time = 0.0

    for i, ex in enumerate(subset):
        qid = ex["id"]
        question = ex["question"]
        golden = ex.get("answer", "")
        context = ex["context"]

        prompt = f"""Read the following context and answer the question. Give ONLY the answer, nothing else. No explanation, no reasoning, just the answer entity.

Context:
{context}

Question: {question}

Answer:"""

        print(f"Q{i+1}/{len(subset)}: {question}")
        print(f"  Gold: {golden}")

        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            elapsed = time.time() - t0
            answer = response.choices[0].message.content.strip()
            input_toks = response.usage.prompt_tokens
            output_toks = response.usage.completion_tokens

            total_input_tokens += input_toks
            total_output_tokens += output_toks
            total_time += elapsed

            cost = (input_toks * prices["input"] / 1_000_000) + (output_toks * prices["output"] / 1_000_000)

            final_score = _score(answer, golden)
            hit = bool(golden and golden.lower() in answer.lower())

            print(f"  Answer: {answer}")
            print(f"  Match: {'PASS' if hit else 'FAIL'}  Score: {final_score:.2f}")
            print(f"  Time: {elapsed:.2f}s  Tokens: {input_toks}+{output_toks}  Cost: ${cost:.4f}")
            print()

            results.append({
                "id": qid, "hops": ex.get("num_hops"),
                "question": question, "golden": golden,
                "answer": answer, "hit": hit,
                "final_score": final_score,
                "time_sec": elapsed,
                "input_tokens": input_toks,
                "output_tokens": output_toks,
                "cost_usd": cost,
            })
        except Exception as e:
            elapsed = time.time() - t0
            total_time += elapsed
            print(f"  ERROR: {e}")
            print()
            results.append({
                "id": qid, "hops": ex.get("num_hops"),
                "question": question, "golden": golden,
                "answer": "(error)", "hit": False,
                "final_score": 0.0,
                "time_sec": elapsed,
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            })

    # --- AGGREGATE ---
    n = len(results)
    hits = sum(1 for r in results if r["hit"])
    avg_score = sum(r["final_score"] for r in results) / n if n else 0
    avg_time = total_time / n if n else 0
    total_cost = sum(r["cost_usd"] for r in results)
    avg_cost = total_cost / n if n else 0

    print(f"\n{'#' * 60}")
    print(f"# GPT-4 BASELINE RESULTS ({args.model})")
    print(f"{'#' * 60}")
    print(f"Questions run:      {n}")
    print(f"Exact matches:      {hits}/{n}  ({100*hits/n:.1f}%)")
    print(f"Avg final score:    {avg_score:.3f}")
    print(f"Avg time/question:  {avg_time:.2f}s")
    print(f"Total tokens:       {total_input_tokens} input + {total_output_tokens} output")
    print(f"Total cost:         ${total_cost:.4f}")
    print(f"Avg cost/question:  ${avg_cost:.4f}")
    print()
    print(f"Per-question:")
    for r in results:
        tag = "PASS" if r["hit"] else "FAIL"
        print(f"  [{tag}] {r['id']:<40s} score={r['final_score']:.2f}  time={r['time_sec']:.2f}s  cost=${r['cost_usd']:.4f}")
    print(f"{'#' * 60}")

    # Save JSON
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(log_dir, f"gpt4_baseline_{args.model}_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {json_path}")

if __name__ == "__main__":
    main()
