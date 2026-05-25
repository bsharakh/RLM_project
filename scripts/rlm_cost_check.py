#!/usr/bin/env python
"""Measure real RLM cost per question by tracking all API calls."""
import argparse, os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from src.rlm import RLM, extract_answer
from src.runner import run_question
from tests.musique_loader import load_musique_dataset

# gpt-4o-mini pricing per 1M tokens
PRICE_INPUT = 0.15
PRICE_OUTPUT = 0.60

class TrackedCaller:
    """Wraps OpenAI calls to track token usage."""
    def __init__(self, client, model):
        self.client = client
        self.model = model
        self.total_input = 0
        self.total_output = 0
        self.call_count = 0

    def __call__(self, prompt):
        self.call_count += 1
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=2048,
        )
        self.total_input += r.usage.prompt_tokens
        self.total_output += r.usage.completion_tokens
        return r.choices[0].message.content

    def reset(self):
        self.total_input = 0
        self.total_output = 0
        self.call_count = 0

    def cost(self):
        return (self.total_input * PRICE_INPUT / 1_000_000) + (self.total_output * PRICE_OUTPUT / 1_000_000)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    ds = load_musique_dataset(split="validation", answerable_only=True)
    if args.hops:
        ds = [ex for ex in ds if ex.get("num_hops") == args.hops]
        print(f"Filtered to {len(ds)} {args.hops}-hop questions")

    subset = ds[args.idx : args.idx + args.num]
    print(f"Running {len(subset)} questions with RLM cost tracking\n")

    client = OpenAI()
    boss_caller = TrackedCaller(client, args.model)
    intern_caller = TrackedCaller(client, args.model)
    rlm = RLM(boss_caller, intern_caller)

    results = []
    grand_total_cost = 0.0
    grand_total_time = 0.0

    for i, ex in enumerate(subset):
        qid = ex["id"]
        question = ex["question"]
        golden = ex.get("answer", "")
        chunks = ex["context"].split("\n\n")

        # Reset counters for this question
        boss_caller.reset()
        intern_caller.reset()

        print(f"Q{i+1}/{len(subset)}: {question}")
        print(f"  Gold: {golden}")

        t0 = time.time()
        try:
            answer, info = run_question(rlm, question, chunks, golden=golden)
            elapsed = time.time() - t0

            q_cost = boss_caller.cost() + intern_caller.cost()
            q_input = boss_caller.total_input + intern_caller.total_input
            q_output = boss_caller.total_output + intern_caller.total_output
            q_calls = boss_caller.call_count + intern_caller.call_count

            grand_total_cost += q_cost
            grand_total_time += elapsed

            print(f"  Answer: {answer}")
            print(f"  Match: {'PASS' if info['hit'] else 'FAIL'}  Score: {info['final_score']:.2f}")
            print(f"  Time: {elapsed:.2f}s  Calls: {q_calls}  Tokens: {q_input}+{q_output}  Cost: ${q_cost:.4f}")
            print()

            results.append({
                "id": qid, "question": question, "golden": golden,
                "answer": answer, "hit": info["hit"],
                "final_score": info["final_score"],
                "time_sec": elapsed,
                "api_calls": q_calls,
                "input_tokens": q_input,
                "output_tokens": q_output,
                "cost_usd": q_cost,
                "steps": info["steps"],
            })
        except Exception as e:
            elapsed = time.time() - t0
            grand_total_time += elapsed
            print(f"  ERROR: {e}")
            print()
            results.append({
                "id": qid, "question": question, "golden": golden,
                "answer": "(error)", "hit": False,
                "final_score": 0.0,
                "time_sec": elapsed, "api_calls": 0,
                "input_tokens": 0, "output_tokens": 0,
                "cost_usd": 0.0, "steps": 0,
            })

    # --- AGGREGATE ---
    n = len(results)
    hits = sum(1 for r in results if r["hit"])
    avg_cost = grand_total_cost / n if n else 0
    avg_time = grand_total_time / n if n else 0
    avg_calls = sum(r["api_calls"] for r in results) / n if n else 0
    avg_input = sum(r["input_tokens"] for r in results) / n if n else 0
    avg_output = sum(r["output_tokens"] for r in results) / n if n else 0

    print(f"\n{'#' * 60}")
    print(f"# RLM COST MEASUREMENT ({args.model})")
    print(f"{'#' * 60}")
    print(f"Questions run:       {n}")
    print(f"Exact matches:       {hits}/{n}  ({100*hits/n:.1f}%)")
    print(f"Avg time/question:   {avg_time:.2f}s")
    print(f"Avg API calls/q:     {avg_calls:.1f}")
    print(f"Avg tokens/q:        {avg_input:.0f} input + {avg_output:.0f} output")
    print(f"Avg cost/question:   ${avg_cost:.4f}")
    print(f"Total cost:          ${grand_total_cost:.4f}")
    print()
    print(f"Per-question:")
    for r in results:
        tag = "PASS" if r["hit"] else "FAIL"
        print(f"  [{tag}] {r['id']:<40s} calls={r['api_calls']}  tokens={r['input_tokens']}+{r['output_tokens']}  cost=${r['cost_usd']:.4f}  time={r['time_sec']:.1f}s")
    print(f"{'#' * 60}")

    # Save JSON
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(log_dir, f"rlm_cost_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {json_path}")

if __name__ == "__main__":
    main()
