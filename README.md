# Recursive Language Model for Multi-Hop Question Answering

A privacy-preserving two-role architecture for multi-hop question answering, evaluated on the [MuSiQue](https://github.com/StonyBrookNLP/musique) benchmark.

**BSc Final Year Project** — Department of Computer Science, University of Haifa, 2026

## Overview

This project implements a Recursive Language Model (RLM) that answers multi-hop questions by reading document chunks one at a time, rather than sending the full document to a large language model. The system uses two roles:

- **Intern**: A stateless fact extractor that reads one chunk and reports relevant facts. Never sees the full document.
- **Boss**: A reasoning agent that maintains a unified narrative, composes multi-hop chains from reported facts, and tracks a running best answer. Never sees raw document chunks.

This architecture enables **privacy-preserving question answering** — the intern and boss can run on any local LLM, and no chunk of the document ever needs to leave the organization's infrastructure.

## Results

| System | Dataset | Exact Match | Cost/Question | Time/Question |
|--------|---------|-------------|---------------|---------------|
| GPT-4o (full context) | 2-hop, n=100 | **55.0%** | $0.0054 | 3.7s |
| RLM (gpt-4o-mini) | 2-hop, n=500 | **36.8%** | $0.0027 | 55.5s |
| RLM (gpt-4o-mini) | 3-hop, n=100 | **15.0%** | ~$0.003 | ~55s |
| RLM (gpt-4o-mini) | 4-hop, n=100 | **25.0%** | ~$0.003 | ~55s |

The RLM achieves **67% of GPT-4o's accuracy at 50% of the cost**, and is **25% more cost-efficient per correct answer**.

## Architecture

```
Question + Document
        ↓
   Chunk Splitter → One chunk at a time
        ↓
┌──────────────────────────────┐
│  INTERN (stateless)          │
│  Input: question + one chunk │
│  Output: extracted facts     │
│  ⚠ Never sees full document │
└──────────────────────────────┘
        ↓ extracted facts only
┌──────────────────────────────┐
│  BOSS (maintains reasoning)  │
│  Updates reasoning narrative │
│  Composes multi-hop chains   │
│  Output: CURRENT BEST ANSWER │
│  ⚠ Never sees raw chunks    │
└──────────────────────────────┘
        ↓
   Reasoning Trimming + Answer Guard
        ↓
   Chunk Selector (keyword scoring)
        ↓
   ↺ Loop repeats until all chunks read
        ↓
   FINAL ANSWER
```

**Privacy constraint**: Sequential access only. No document preprocessing.

## Key Features

- **Unified reasoning narrative**: The boss thinks out loud rather than maintaining structured state, improving accuracy from ~9% to 36.8%.
- **Reasoning trimming**: Keeps the boss context under 15 lines to prevent context bloat in small models.
- **Answer guard**: Code-level protection against answer regression — once a good answer is found, it cannot be dropped to "none yet."
- **Keyword-based chunk selection**: Steers retrieval toward relevant chunks as entity names are discovered.

## Setup

### Requirements

- Python 3.9+
- OpenAI API key

### Installation

```bash
git clone https://github.com/bsharakh/RLM_project.git
cd RLM_project
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

## Usage

### Run RLM on MuSiQue

```bash
# 5-question smoke test
python scripts/run_musique.py --hops 2 --num 5

# Full 2-hop evaluation (500 questions)
python scripts/run_musique.py --hops 2 --num 500 > logs/rlm_2hop_500.txt

# 3-hop and 4-hop evaluation
python scripts/run_musique.py --hops 3 --num 100 > logs/rlm_3hop_100.txt
python scripts/run_musique.py --hops 4 --num 100 > logs/rlm_4hop_100.txt
```

### Run GPT-4o Baseline

```bash
python scripts/gpt4_baseline.py --hops 2 --num 100 --model gpt-4o > logs/baseline_gpt4o.txt
```

### Measure RLM Cost

```bash
python scripts/rlm_cost_check.py --hops 2 --num 10 > logs/rlm_cost.txt
```

## Project Structure

```
├── config/
│   └── prompts.py           # Intern and Boss prompt templates
├── src/
│   ├── rlm.py               # RLM class, extract_answer, reasoning trimming, answer guard
│   └── runner.py             # Question runner with scoring and step tracking
├── scripts/
│   ├── run_musique.py        # Batch evaluation on MuSiQue
│   ├── gpt4_baseline.py      # GPT-4o full-context baseline
│   └── rlm_cost_check.py     # Cost measurement with token tracking
├── tests/
│   └── musique_loader.py     # HuggingFace MuSiQue dataset loader
└── logs/                     # Output logs and JSON results
```

## Failure Modes

We identify four principal failure modes through trace analysis:

1. **Alias Resolution**: The boss cannot connect different surface forms of the same entity across chunks (e.g., "German Aerospace Center" vs "DLR").
2. **Answer Regression (DROPPED)**: The boss replaces a correct answer with a wrong one from a later chunk. Affects 4.2% of 2-hop questions.
3. **Intern Hallucination**: The intern reports facts from its training data rather than from the chunk text.
4. **Retrieval Order Sensitivity**: Keyword scoring is weak in early steps when no entities have been discovered yet.

## Citation

If you use this work, please cite:

```
@misc{khoury2026rlm,
  title={Recursive Language Model for Multi-Hop Question Answering},
  author={Khoury, Bshara},
  year={2026},
  institution={University of Haifa}
}
```

## License

This project is part of a BSc thesis at the University of Haifa.
