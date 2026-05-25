#!/usr/bin/env python
"""Run RLM on MuSiQue questions with full statistics."""
import argparse, os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from src.rlm import RLM, extract_answer
from src.runner import run_question
from tests.musique_loader import load_musique_dataset

def _build_openai_caller(model="gpt-4o-mini"):
    client = OpenAI()
    def caller(prompt):
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=2048,
        )
        return r.choices[0].message.content
    return caller

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--model", default=os.environ.get("RLM_MODEL", "gpt-4o-mini"))
    args = ap.parse_args()

    ds = load_musique_dataset(split="validation", answerable_only=True)
    if args.hops:
        ds = [ex for ex in ds if ex.get("num_hops") == args.hops]
        print(f"Filtered to {len(ds)} {args.hops}-hop questions")

    subset = ds[args.idx : args.idx + args.num]
    print(f"Running {len(subset)} questions (idx {args.idx}..{args.idx + len(subset) - 1})\n")

    boss_llm = _build_openai_caller(args.model)
    intern_llm = _build_openai_caller(args.model)
    rlm = RLM(boss_llm, intern_llm)

    results = []

    for i, ex in enumerate(subset):
        qid = ex["id"]
        question = ex["question"]
        golden = ex.get("answer", "")
        chunks = ex["context"].split("\n\n")

        print(f"\n{'#' * 60}")
        print(f"# QUESTION {i+1}/{len(subset)}  (id={qid}, hops={ex.get('num_hops','?')})")
        print(f"{'#' * 60}")
        print(f"Q: {question}")
        print(f"Gold: {golden}\n")

        try:
            answer, info = run_question(rlm, question, chunks, golden=golden)
            results.append({
                "id": qid, "hops": ex.get("num_hops"),
                "question": question, "golden": golden,
                "answer": answer, "hit": info["hit"],
                "final_score": info["final_score"],
                "best_score": info["best_score"],
                "best_step": info["best_step"],
                "steps": info["steps"],
                "total_chunks": len(chunks),
            })
        except Exception as e:
            print(f"ERROR on {qid}: {e}")
            results.append({
                "id": qid, "hops": ex.get("num_hops"),
                "question": question, "golden": golden,
                "answer": "(error)", "hit": False,
                "final_score": 0.0, "best_score": 0.0,
                "best_step": 0, "steps": 0,
                "total_chunks": len(chunks),
            })

    # -- AGGREGATE --
    n = len(results)
    hits = sum(1 for r in results if r["hit"])
    avg_final = sum(r["final_score"] for r in results) / n if n else 0
    avg_best = sum(r["best_score"] for r in results) / n if n else 0
    avg_steps = sum(r["steps"] for r in results) / n if n else 0

    answered_correct = sum(1 for r in results if r["hit"])
    answered_wrong = sum(1 for r in results if not r["hit"] and r["answer"] not in ["(no answer)", "(error)", "(none yet)", ""])
    no_answer = sum(1 for r in results if r["answer"] in ["(no answer)", "(error)", "(none yet)", ""])
    dropped_answer = sum(1 for r in results if not r["hit"] and r["best_score"] > 0.5 and r["final_score"] < 0.5)

    print(f"\n{'#' * 60}")
    print(f"# AGGREGATE RESULTS")
    print(f"{'#' * 60}")
    print(f"Questions run:    {n}")
    print(f"Exact matches:    {hits}/{n}  ({100*hits/n:.1f}%)")
    print(f"Avg final score:  {avg_final:.3f}")
    print(f"Avg best score:   {avg_best:.3f}")
    print(f"Avg steps used:   {avg_steps:.1f}")
    print()
    print(f"--- Answer Categories ---")
    print(f"  Answered correctly:     {answered_correct}/{n}  ({100*answered_correct/n:.1f}%)")
    print(f"  Answered incorrectly:   {answered_wrong}/{n}  ({100*answered_wrong/n:.1f}%)")
    print(f"  No answer produced:     {no_answer}/{n}  ({100*no_answer/n:.1f}%)")
    print(f"  Had correct then lost:  {dropped_answer}/{n}  ({100*dropped_answer/n:.1f}%)")
    print()

    print(f"Per-question:")
    for r in results:
        tag = "PASS" if r["hit"] else "FAIL"
        dropped = " [DROPPED]" if (not r["hit"] and r["best_score"] > 0.5 and r["final_score"] < 0.5) else ""
        best_info = f"  best@step={r['best_step']}/{r['steps']}" if r["best_score"] > 0 else ""
        print(f"  [{tag}] {r['id']:<40s} hops={r['hops']}  score={r['final_score']:.2f}  best={r['best_score']:.2f}  steps={r['steps']}{best_info}{dropped}")
    print(f"{'#' * 60}")

    # -- Save JSON --
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(log_dir, f"batch_results_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {json_path}")

if __name__ == "__main__":
    main()