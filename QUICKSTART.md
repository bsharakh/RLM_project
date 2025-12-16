# Quick Start Guide

## Setup (5 minutes)

1. **Install dependencies:**
   ```bash
   cd rlm-project
   pip install -r requirements.txt --break-system-packages
   ```

2. **Configure API key:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

3. **Test the installation:**
   ```bash
   python tests/test_runner.py
   ```

## Usage Examples

### Example 1: Interactive REPL

```bash
python src/repl.py
```

Then in the REPL:
```python
> context = "Python was created by Guido van Rossum in 1991. It is known for its simple syntax and readability. Python is used in web development, data science, and automation."

> ask("Who created Python?")
# Watch the RLM and SubLLM interact!

> ask("What is Python used for?")
```

### Example 2: Run Test Suite

```bash
python tests/test_runner.py
```

This will run all test questions and show you the iterations.

### Example 3: Programmatic Usage

```python
from src.rlm import RLM

# Initialize
rlm = RLM(max_iterations=5)

# Your context
context = """
TechCorp was founded in 2010 by Jane Smith.
The company focuses on AI solutions.
In 2023, they launched their flagship product.
"""

# Ask a question
answer = rlm.answer("Who founded TechCorp?", context)
print(answer)
```

## Understanding the Output

When you run a query, you'll see:

1. **Blue section**: New question starting
2. **Yellow sections**: Each iteration (Boss → Intern)
   - Boss's question in cyan
   - Intern's answer in green
3. **Magenta section**: Final answer

## Logs

All interactions are logged to `logs/rlm_session_TIMESTAMP.json`

You can review:
- Every question the boss asked
- Every answer the intern gave
- The final answer
- Timestamps and metadata

## Customization

Edit `.env` to change:
- `RLM_MODEL` - The boss model (default: gpt-4)
- `SUBLM_MODEL` - The intern model (default: gpt-4o-mini)
- `MAX_ITERATIONS` - How many times to iterate (default: 5)
- `MAX_DEPTH` - Recursion depth (default: 1)

## Tips

1. **Start simple**: Test with clear, factual questions first
2. **Check logs**: Review the logs to see how the system thinks
3. **Adjust iterations**: Complex questions may need more iterations
4. **Monitor costs**: The boss model (GPT-4) is more expensive than the intern (GPT-4o-mini)

## Next Steps

Once this works well, you can:
1. Add vector database integration
2. Implement caching for common questions
3. Add evaluation metrics
4. Build a web interface
5. Scale to production workloads
