"""Run RLM, print step-by-step reasoning, score against golden."""
import re
from src.rlm import extract_answer

_WORD = re.compile(r"[A-Za-z0-9]+")
def _toks(s): return {w.lower() for w in _WORD.findall(s or "")}

def _score(answer, golden):
    if not golden: return 0.0
    g = _toks(golden)
    if not g: return 0.0
    return len(_toks(answer) & g) / len(g)


def run_question(rlm, question, chunks, golden=None, max_iters=None):
    if max_iters is None:
        max_iters = len(chunks)

    state = {"steps": 0, "best_score": 0.0, "best_step": 0}

    def on_step(step, chunk_idx, finding, reasoning):
        state["steps"] = step
        committed = extract_answer(reasoning)
        score = _score(committed, golden) if committed else 0.0
        if score > state["best_score"]:
            state["best_score"] = score
            state["best_step"] = step
        snippet = (finding or "").replace("\n", " ")[:180]
        print(f"\n-- step {step} | chunk {chunk_idx} | score {score:.2f} --")
        print(f"intern: {snippet}")
        print(f"reasoning:\n{reasoning}")

    final_reasoning = rlm.answer(question, chunks, max_iters=max_iters, on_step=on_step)
    final_answer = extract_answer(final_reasoning) or "(no answer)"
    final_score = _score(final_answer, golden)
    hit = bool(golden and golden.lower() in (final_answer or "").lower())

    print("\n" + "=" * 60)
    print(f"QUESTION:     {question}")
    print(f"FINAL ANSWER: {final_answer}")
    if golden:
        print(f"GOLDEN:       {golden}")
        print(f"MATCH:        {'PASS' if hit else 'FAIL'}")
        print(f"FINAL SCORE:  {final_score:.2f}")
        print(f"BEST SCORE:   {state['best_score']:.2f}")
        print(f"BEST STEP:    {state['best_step']}")
    print(f"STEPS:        {state['steps']}")
    print(f"CHUNKS SEEN:  {state['steps']} / {len(chunks)}")
    print("=" * 60)

    return final_answer, {
        "steps": state["steps"],
        "best_score": state["best_score"],
        "best_step": state["best_step"],
        "final_score": final_score,
        "hit": hit,
    }