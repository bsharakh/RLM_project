# RLM (Recursive Language Model) Project

A recursive language model system where a root LLM (boss) delegates context reading to a smaller sub-LLM (intern) iteratively until finding the optimal answer.

## Architecture

```
RLM (Boss)
├── Formulates high-level questions
├── Evaluates answers from SubLLM
└── Iterates until satisfied

SubLLM (Intern)
├── Reads context from REPL environment
├── Answers specific questions
└── Returns findings to RLM
```

## Features

- ✅ Recursive question-answering with configurable depth
- ✅ Boss-intern model with iterative refinement
- ✅ REPL environment with context variable
- ✅ Environment-based model configuration
- ✅ Logging system for full iteration tracking
- ✅ Pluggable test questions system

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt --break-system-packages
```

2. Configure your API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

3. Run the REPL:
```bash
python src/repl.py
```

## Project Structure

```
rlm-project/
├── src/
│   ├── rlm.py           # Main RLM (boss) logic
│   ├── sublm.py         # SubLLM (intern) logic
│   ├── repl.py          # Interactive REPL environment
│   ├── logger.py        # Logging system
│   └── config.py        # Configuration loader
├── tests/
│   └── test_questions.json  # Pluggable test questions
├── config/
│   └── prompts.py       # System prompts
├── logs/               # Iteration logs (auto-generated)
├── .env               # Environment variables
└── requirements.txt   # Python dependencies
```

## Usage

### Interactive REPL

```python
# Start REPL
python src/repl.py

# Set context
> context = "Your knowledge base here..."

# Ask a question
> ask("What is the main topic?")
```

### Running Test Questions

```python
from src.rlm import RLM
from tests.test_questions import load_questions

rlm = RLM(max_depth=3)
questions = load_questions()

for q in questions:
    answer = rlm.answer(q["question"], q["context"])
    print(f"Q: {q['question']}\nA: {answer}\n")
```

## Configuration

Edit `.env`:
```
OPENAI_API_KEY=your_key_here
RLM_MODEL=gpt-4  # Boss model
SUBLM_MODEL=gpt-4o-mini  # Intern model
MAX_ITERATIONS=5
MAX_DEPTH=1
ENABLE_LOGGING=true
LOG_LEVEL=INFO
```

## Extending the Project

This is designed as a foundation. Next steps:
1. ✅ Add test question system
2. ✅ Implement detailed logging
3. ✅ Add metrics and evaluation
4. ?

## License

MIT
