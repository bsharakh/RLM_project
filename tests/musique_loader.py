"""
MuSiQue Dataset Loader for RLM
==============================

Loads the MuSiQue multi-hop QA dataset (StonyBrookNLP/musique, TACL 2022) and
exposes it in the same shape that `tests/test_musique.py` expects:

    - load_musique_dataset(split, answerable_only)   -> List[dict]
    - get_musique_by_hops(questions, num_hops)       -> List[dict]
    - get_musique_sample(questions, n)               -> List[dict]
    - build_shared_pool(questions, pool_size, ...)   -> List[dict]

Each question dict has the keys consumed by test_musique.py and src/rlm.py:

    {
      'id':                    str,
      'question':              str,
      'answer':                str,
      'context':               str,           # all paragraphs concatenated
      'num_hops':              int,           # 2, 3, or 4
      'answerable':            bool,
      'question_decomposition': [...],
      'supporting_paragraph_idxs': [int, ...],  # indices of gold paragraphs
      'context_chars':         int,
      'num_paragraphs':        int,
      'num_supporting':        int,
    }

Shared-pool runs (for haystack experiments)
-------------------------------------------
`build_shared_pool` takes N questions, concatenates ALL their paragraphs into
ONE shared corpus, and returns N question dicts that all reference that
*identical* corpus (same bytes for every question — no per-question rotation
or manipulation).

The paragraphs are globally shuffled exactly once when the pool is built,
using the pool seed, so where each question's needles end up is arbitrary —
some early, some late, some scattered, some clustered. This simulates the
real-world case where the answer could live anywhere in the corpus.

Loading strategy (in order):
    1. Local file at  data/musique/musique_ans_v1.0_<split>.jsonl
    2. HuggingFace `datasets` -> StonyBrookNLP/musique
    3. Helpful error message
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data" / "musique"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_musique_dataset(
    split: str = "validation",
    answerable_only: bool = True,
    max_questions: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load MuSiQue questions in the RLM-compatible shape."""
    raw_examples = _load_raw(split=split)

    converted: List[Dict[str, Any]] = []
    for ex in raw_examples:
        q = _convert_example(ex)
        if q is None:
            continue
        if answerable_only and not q["answerable"]:
            continue
        converted.append(q)
        if max_questions and len(converted) >= max_questions:
            break

    print(f"✅ Loaded {len(converted)} MuSiQue questions "
          f"(split={split}, answerable_only={answerable_only})")
    return converted


def get_musique_by_hops(
    questions: List[Dict[str, Any]],
    num_hops: int,
) -> List[Dict[str, Any]]:
    """Filter questions by exact number of hops (2, 3, or 4)."""
    return [q for q in questions if q.get("num_hops") == num_hops]


def get_musique_sample(
    questions: List[Dict[str, Any]],
    n: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Return up to `n` questions, deterministic via `seed`."""
    if n >= len(questions):
        return list(questions)
    rng = random.Random(seed)
    return rng.sample(questions, n)


# ---------------------------------------------------------------------------
# Shared-pool builder
# ---------------------------------------------------------------------------

def build_shared_pool(
    questions: List[Dict[str, Any]],
    pool_size: int,
    pool_name: str = "pool",
    fixed_question_ids: Optional[List[str]] = None,
    hop_balance: bool = True,
    seed: int = 7,
) -> List[Dict[str, Any]]:
    """
    Build a shared-corpus run from a pool of MuSiQue questions.

    All N questions in the pool share the SAME underlying paragraph set
    (everyone's paragraphs concatenated into one corpus). The paragraphs are
    globally shuffled once at pool build time using `seed`, so each
    question's needles end up wherever the shuffle puts them — could be
    early, late, scattered, or clustered. No per-question manipulation.

    Args:
        questions:          The full pool of converted MuSiQue questions to
                            draw from (output of load_musique_dataset).
        pool_size:          How many questions to put in this pool.
        pool_name:          Tag stored on each returned question dict.
        fixed_question_ids: Optional list of question ids that MUST be in
                            this pool (used to share questions across pools).
        hop_balance:        If True, try to pick a mix of 2/3/4-hop questions.
        seed:               RNG seed for selection AND for the global
                            paragraph shuffle.

    Returns:
        List of `pool_size` question dicts. The 'context' string is BYTE
        IDENTICAL across every question in the pool. The
        'supporting_paragraph_idxs' field tells you where each question's
        needles ended up in that shared shuffled corpus.
    """
    selected = _select_pool_questions(
        questions=questions,
        pool_size=pool_size,
        fixed_question_ids=fixed_question_ids or [],
        hop_balance=hop_balance,
        seed=seed,
    )

    # Build the canonical shared paragraph list. Each entry remembers which
    # question(s) it supports so we can compute per-question needle indices
    # after the global shuffle.
    canonical_blocks: List[Dict[str, Any]] = []
    for q in selected:
        for b in q["_raw_blocks"]:
            canonical_blocks.append({
                "text": b["text"],
                "is_supporting_for": q["id"] if b["is_supporting"] else None,
                "owner_id": q["id"],
            })

    # Shuffle ONCE, deterministically. After this, paragraph order is
    # arbitrary — the position of each question's needles is whatever the
    # shuffle happened to produce. No rotation, no per-question reordering.
    rng = random.Random(seed)
    rng.shuffle(canonical_blocks)

    n_blocks = len(canonical_blocks)
    pool_member_ids = [q["id"] for q in selected]

    # Build the single shared context string. Re-number paragraph indices
    # in the shuffled order so the [N] prefixes are sequential.
    shared_blocks_text = [
        f"[{i}] {b['text']}" for i, b in enumerate(canonical_blocks)
    ]
    shared_context = "\n\n".join(shared_blocks_text)

    # Per question: figure out which positions in the shared corpus hold
    # this question's supporting paragraphs.
    out_questions: List[Dict[str, Any]] = []
    for q in selected:
        supporting_idxs = [
            i for i, b in enumerate(canonical_blocks)
            if b["is_supporting_for"] == q["id"]
        ]
        needle_pct = (
            sum(supporting_idxs) / (len(supporting_idxs) * n_blocks) * 100
            if supporting_idxs else None
        )

        out_questions.append({
            "id": q["id"],
            "question": q["question"],
            "answer": q["answer"],
            "answer_aliases": q["answer_aliases"],
            "context": shared_context,            # IDENTICAL across all
            "context_chars": len(shared_context),
            "num_paragraphs": n_blocks,
            "num_supporting": len(supporting_idxs),
            "num_hops": q["num_hops"],
            "answerable": q["answerable"],
            "question_decomposition": q["question_decomposition"],
            "supporting_paragraph_idxs": supporting_idxs,
            # Pool metadata
            "pool_name": pool_name,
            "pool_size": pool_size,
            "pool_member_ids": pool_member_ids,
            "needle_position_pct": needle_pct,
        })

    print(
        f"📦 Built shared pool '{pool_name}': "
        f"{pool_size} questions, "
        f"{n_blocks} paragraphs, "
        f"{len(shared_context):,} chars  "
        f"(globally shuffled, seed={seed})"
    )
    return out_questions


def _select_pool_questions(
    questions: List[Dict[str, Any]],
    pool_size: int,
    fixed_question_ids: List[str],
    hop_balance: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    """Pick `pool_size` questions, honoring fixed ids and hop balance."""
    by_id = {q["id"]: q for q in questions}

    chosen: List[Dict[str, Any]] = []
    chosen_ids = set()
    for qid in fixed_question_ids:
        if qid in by_id and qid not in chosen_ids:
            chosen.append(by_id[qid])
            chosen_ids.add(qid)
        else:
            print(f"⚠️  fixed_question_ids: id '{qid}' not found in pool, skipping")

    remaining_slots = pool_size - len(chosen)
    if remaining_slots <= 0:
        return chosen[:pool_size]

    rng = random.Random(seed)
    pool = [q for q in questions if q["id"] not in chosen_ids]

    if hop_balance:
        buckets = {h: [q for q in pool if q["num_hops"] == h] for h in (2, 3, 4)}
        for h in buckets:
            rng.shuffle(buckets[h])

        hop_order = [2, 3, 4]
        i = 0
        while remaining_slots > 0 and any(buckets[h] for h in hop_order):
            h = hop_order[i % len(hop_order)]
            i += 1
            if buckets[h]:
                chosen.append(buckets[h].pop())
                remaining_slots -= 1

    if remaining_slots > 0:
        leftover = [q for q in pool if q["id"] not in {c["id"] for c in chosen}]
        rng.shuffle(leftover)
        chosen.extend(leftover[:remaining_slots])

    return chosen[:pool_size]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_raw(split: str) -> List[Dict[str, Any]]:
    """Try local file first, then HuggingFace, then raise with instructions."""
    examples = _load_from_local(split)
    if examples:
        print(f"📂 Loaded {len(examples)} raw examples from local file")
        return examples

    examples = _load_from_huggingface(split)
    if examples:
        print(f"🤗 Loaded {len(examples)} raw examples from HuggingFace")
        return examples

    raise FileNotFoundError(
        "Could not find MuSiQue data.\n"
        "Please do ONE of:\n"
        "  1. Download musique_ans_v1.0_dev.jsonl (or _train.jsonl) from\n"
        "     https://github.com/StonyBrookNLP/musique\n"
        f"     and put it in: {_DATA_DIR}\n"
        "  2. Or:  pip install datasets   (so HuggingFace fallback works)\n"
    )


def _load_from_local(split: str) -> Optional[List[Dict[str, Any]]]:
    if not _DATA_DIR.exists():
        return None

    aliases = {split}
    if split == "validation":
        aliases.add("dev")
    if split == "dev":
        aliases.add("validation")

    candidates = []
    for path in _DATA_DIR.glob("*.jsonl"):
        name = path.name.lower()
        if any(a in name for a in aliases):
            candidates.append(path)

    if not candidates:
        return None

    candidates.sort(key=lambda p: ("ans" not in p.name, len(p.name)))
    chosen = candidates[0]

    examples = []
    with open(chosen, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _load_from_huggingface(split: str) -> Optional[List[Dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError:
        return None

    hf_split = "validation" if split in ("validation", "dev") else split

    attempts = [
        ("StonyBrookNLP/musique", None),
        ("StonyBrookNLP/musique", "musique_ans_v1.0"),
        ("dgslibisey/MuSiQue", None),
    ]
    for repo, config in attempts:
        try:
            if config:
                ds = load_dataset(repo, config, split=hf_split)
            else:
                ds = load_dataset(repo, split=hf_split)
            return [dict(ex) for ex in ds]
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Conversion: raw MuSiQue example -> RLM-compatible question dict
# ---------------------------------------------------------------------------

def _convert_example(ex: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one raw MuSiQue example into the RLM input shape."""
    try:
        paragraphs = ex.get("paragraphs", []) or []
        if not paragraphs:
            return None

        raw_blocks = []
        context_parts = []
        supporting_idxs = []
        for p in paragraphs:
            idx = p.get("idx", len(context_parts))
            title = p.get("title", "") or ""
            text = p.get("paragraph_text", "") or ""
            block_text = f"{title}\n{text}".strip() if title else text.strip()
            raw_blocks.append({
                "text": block_text,
                "is_supporting": bool(p.get("is_supporting")),
            })
            context_parts.append(f"[{idx}] {block_text}")
            if p.get("is_supporting"):
                supporting_idxs.append(idx)
        context = "\n\n".join(context_parts)

        decomposition = []
        for hop in ex.get("question_decomposition", []) or []:
            decomposition.append({
                "question": hop.get("question", ""),
                "answer": hop.get("answer", ""),
                "paragraph_support_idx": hop.get("paragraph_support_idx"),
            })

        ex_id = ex.get("id", "")
        num_hops = _infer_num_hops(ex_id, decomposition)

        return {
            "id": ex_id,
            "question": ex.get("question", ""),
            "answer": ex.get("answer", ""),
            "answer_aliases": ex.get("answer_aliases", []) or [],
            "context": context,
            "context_chars": len(context),
            "num_paragraphs": len(raw_blocks),
            "num_supporting": len(supporting_idxs),
            "num_hops": num_hops,
            "answerable": bool(ex.get("answerable", True)),
            "question_decomposition": decomposition,
            "supporting_paragraph_idxs": supporting_idxs,
            "_raw_blocks": raw_blocks,  # internal — used by build_shared_pool
        }
    except Exception as e:
        print(f"⚠️  Skipping malformed example: {e}")
        return None


def _infer_num_hops(ex_id: str, decomposition: List[Dict[str, Any]]) -> int:
    if isinstance(ex_id, str):
        for prefix in ("2hop", "3hop", "4hop"):
            if ex_id.startswith(prefix):
                return int(prefix[0])
    if decomposition:
        return len(decomposition)
    return 2


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    qs = load_musique_dataset(split="validation", answerable_only=True)
    print(f"\nTotal: {len(qs)}")
    for h in (2, 3, 4):
        print(f"  {h}-hop: {len(get_musique_by_hops(qs, h))}")

    if qs:
        pool_a = build_shared_pool(qs, pool_size=5, pool_name="A", seed=7)
        shared_ids = [pool_a[0]["id"], pool_a[1]["id"]]
        pool_b = build_shared_pool(
            qs, pool_size=10, pool_name="B",
            fixed_question_ids=shared_ids, seed=7,
        )
        print(f"\nPool A members: {[q['id'] for q in pool_a]}")
        print(f"Pool B members: {[q['id'] for q in pool_b]}")
        print(f"Shared subjects: {sorted(set(p['id'] for p in pool_a) & set(p['id'] for p in pool_b))}")

        # Confirm the shared corpus is identical across all pool members.
        assert all(q['context'] == pool_a[0]['context'] for q in pool_a), \
            "Pool A members should share an identical context!"
        print(f"\n✓ All Pool A members share an identical {pool_a[0]['context_chars']:,}-char corpus")

        print(f"\nNeedle positions in Pool A (where each question's needles ended up):")
        for q in pool_a:
            print(f"  {q['id']}: needles at {q['supporting_paragraph_idxs']} "
                  f"of {q['num_paragraphs']} ({q['needle_position_pct']:.1f}% mean)")