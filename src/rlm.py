"""RLM with unified-reasoning boss. Two roles, intern stateless per chunk."""
import re
from config.prompts import BOSS_UPDATE_PROMPT, INTERN_PROMPT

NO_INFO = "No reasoning yet."
_WORD = re.compile(r"[A-Za-z0-9]+")
_STOP = {"the","and","for","with","from","this","that","when","what","where",
         "did","end","was","are","has","have","who","why","how","which"}

def _tokens(s): return {w.lower() for w in _WORD.findall(s or "") if len(w) > 2}
def _keywords(wa, q): return (_tokens(wa) | _tokens(q)) - _STOP
def _score_chunk(c, kws): return len(_tokens(c) & kws)


def extract_answer(reasoning: str) -> str:
    """Pull the CURRENT BEST ANSWER from the reasoning text.
    Searches from bottom up since the last occurrence is the most recent."""
    if not reasoning:
        return ""
    _skip = {
        "none", "n/a", "(none)", "unknown", "none yet", "(none yet)",
        "(none yet).", "no reasoning yet.", "no reasoning yet",
        "no information found yet.", "no information found yet",
        "",
    }
    lines = reasoning.splitlines()
    # Search from bottom — last CURRENT BEST ANSWER is the most recent
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.lower().startswith("current best answer:"):
            ans = stripped[len("current best answer:"):].strip()
            if ans and ans.lower() not in _skip:
                return ans
    # Fallback: also check for plain "ANSWER:" format
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.lower().startswith("answer:"):
            ans = stripped[len("answer:"):].strip()
            if ans and ans.lower() not in _skip:
                return ans
            if i + 1 < len(lines):
                ans = lines[i + 1].strip()
                if ans and ans.lower() not in _skip:
                    return ans
    return ""


def _trim_reasoning(reasoning, question, max_lines=15):
    """Keep reasoning focused by trimming to most recent and relevant lines."""
    lines = reasoning.strip().splitlines()
    if len(lines) <= max_lines:
        return reasoning
    # Always keep the last line (CURRENT BEST ANSWER) and recent reasoning
    # Score each line by relevance to question + recency
    q_kws = _keywords("", question)
    last_line = lines[-1]
    scorable = lines[:-1]
    scored = []
    for i, line in enumerate(scorable):
        l_kws = _tokens(line)
        relevance = len(l_kws & q_kws)
        recency = i  # higher index = more recent = better
        scored.append((relevance + recency * 0.1, line))
    scored.sort(key=lambda x: -x[0])
    kept = [line for _, line in scored[:max_lines - 1]]
    kept.append(last_line)  # always keep CURRENT BEST ANSWER line
    return "\n".join(kept)


class RLM:
    def __init__(self, llm, intern_llm=None):
        self.llm = llm
        self.intern_llm = intern_llm or llm

    def intern(self, question, chunk, working_answer=""):
        # Intern does NOT see boss reasoning — stays stateless and grounded
        prompt = INTERN_PROMPT.format(
            question=question, context=chunk,
        )
        return self.intern_llm(prompt)

    def boss_update(self, question, working_answer, finding):
        # Trim reasoning to keep boss focused
        trimmed = _trim_reasoning(working_answer, question)
        prompt = BOSS_UPDATE_PROMPT.format(
            question=question,
            working_answer=trimmed,
            finding=finding,
        )
        new = (self.llm(prompt) or "").strip()
        if not new:
            return working_answer
        # Guard 1: never drop a committed answer to nothing
        old_ans = extract_answer(working_answer)
        new_ans = extract_answer(new)
        if old_ans and not new_ans:
            return working_answer
        # Guard 2: never replace an answer with "none yet" or similar
        if old_ans and new_ans and new_ans.lower() in {
            "(none yet)", "none yet", "none", "(none)", "n/a", "unknown",
            "no reasoning yet.", "no reasoning yet",
            "no information found yet.", "no information found yet",
        }:
            return working_answer
        return new

    def _pick_next(self, chunks, explored, working_answer, question, k=2):
        unexplored = [i for i in range(len(chunks)) if i not in explored]
        if not unexplored:
            return []
        kws = _keywords(working_answer, question)
        if kws:
            scored = sorted(unexplored, key=lambda i: (-_score_chunk(chunks[i], kws), i))
            if _score_chunk(chunks[scored[0]], kws) > 0:
                return scored[:k]
        return unexplored[:k]

    def answer(self, question, chunks, max_iters=None, on_step=None):
        # Default: read ALL chunks
        if max_iters is None:
            max_iters = len(chunks)
        return self._answer_with_repl(question, chunks, max_iters, on_step)

    def _answer_with_repl(self, question, chunks, max_iters, on_step=None):
        working_answer = NO_INFO
        explored = set()
        step = 0
        for _ in range(max_iters):
            picks = self._pick_next(chunks, explored, working_answer, question)
            if not picks:
                break
            for idx in picks:
                step += 1
                explored.add(idx)
                finding = self.intern(question, chunks[idx], working_answer)
                working_answer = self.boss_update(question, working_answer, finding)
                if on_step:
                    on_step(step, idx, finding, working_answer)
        return working_answer