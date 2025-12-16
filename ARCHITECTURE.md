# RLM Architecture Documentation

## System Overview

The RLM (Recursive Language Model) system implements a hierarchical question-answering approach where a "boss" model (RLM) delegates information retrieval to an "intern" model (SubLLM) through iterative refinement.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RLM (Boss Model)                            │
│  • Formulates strategic questions                               │
│  • Evaluates answers                                            │
│  • Decides when to stop iterating                               │
│  • Model: GPT-4 (configurable)                                  │
└───────────────────┬───────────────────┬─────────────────────────┘
                    │                   │
          Iteration Loop                │
                    │                   │
                    ▼                   │
┌─────────────────────────────────┐    │
│   SubLLM (Intern Model)         │    │
│  • Reads context/knowledge base │    │
│  • Answers specific questions   │    │
│  • Returns focused responses    │    │
│  • Model: GPT-4o-mini (config)  │    │
└─────────────────┬───────────────┘    │
                  │                     │
                  │   Answer            │
                  └─────────────────────┘
                            │
                            ▼
                    Satisfied? ──No──► Continue Iteration
                            │
                           Yes
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Final Answer                                │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. RLM (rlm.py)
**Role:** Boss/Orchestrator
**Model:** GPT-4 (or configured model)
**Responsibilities:**
- Receives user questions
- Formulates strategic sub-questions for SubLLM
- Evaluates responses for completeness
- Decides when enough information is gathered
- Synthesizes final answer

**Key Methods:**
- `answer(user_question, context, depth)` - Main entry point
- `_formulate_question()` - Creates next question for intern
- `_synthesize_final_answer()` - Creates final response

### 2. SubLLM (sublm.py)
**Role:** Intern/Information Retriever
**Model:** GPT-4o-mini (or configured model)
**Responsibilities:**
- Receives specific questions from RLM
- Reads and analyzes context
- Provides focused, accurate answers
- Only accesses the context (doesn't "know" the original user question)

**Key Methods:**
- `answer(question, context)` - Answers boss's questions

### 3. Logger (logger.py)
**Role:** Tracking and Debugging
**Responsibilities:**
- Logs every iteration
- Tracks questions and answers
- Saves to JSON files
- Provides console output with colors
- Enables debugging and analysis

**Log Structure:**
```json
{
  "session_id": "20241213_143052",
  "logs": [
    {
      "timestamp": "2024-12-13T14:30:52",
      "iteration": 1,
      "type": "RLM_TO_SUBLM",
      "question": "Boss's question",
      "answer": "Intern's answer",
      "metadata": {"depth": 0}
    }
  ]
}
```

### 4. REPL (repl.py)
**Role:** Interactive Interface
**Features:**
- Set context variable
- Ask questions interactively
- View configuration
- Clear context
- Color-coded output

### 5. Test Runner (test_runner.py)
**Role:** Automated Testing
**Features:**
- Load test questions from JSON
- Run multiple test sets
- Compare expected vs actual answers
- Generate detailed logs

## Data Flow

1. **User Input** → User provides question and context
2. **RLM Initialization** → Boss model receives question
3. **Iteration Loop:**
   ```
   a. RLM formulates specific question for intern
   b. SubLLM reads context and answers
   c. RLM evaluates if answer is sufficient
   d. If not sufficient, goto step a
   e. If sufficient, synthesize final answer
   ```
4. **Output** → Return final answer to user
5. **Logging** → All steps logged to file and console

## Configuration System

### Environment Variables (.env)
```
OPENAI_API_KEY=your_key
RLM_MODEL=gpt-4              # Boss model
SUBLM_MODEL=gpt-4o-mini      # Intern model  
MAX_ITERATIONS=5             # Max iteration per question
MAX_DEPTH=1                  # Recursion depth
ENABLE_LOGGING=true
LOG_LEVEL=INFO
```

### Config Loader (config.py)
- Loads environment variables
- Validates configuration
- Provides defaults
- Central configuration access

## Prompt Engineering

### RLM System Prompt
Instructs the boss to:
- Formulate strategic questions
- Build on previous answers
- Signal completion with "FINAL_ANSWER:"
- Be efficient with iterations

### SubLLM System Prompt
Instructs the intern to:
- Read context carefully
- Answer only what's asked
- Be precise and concise
- Acknowledge when info isn't in context

## Iteration Strategy

The system uses an iterative refinement approach:

```
Iteration 1: RLM asks broad question
           SubLLM provides general answer
           
Iteration 2: RLM asks focused follow-up
           SubLLM provides specific details
           
Iteration 3: RLM asks for clarification
           SubLLM provides additional context
           
...until RLM is satisfied or max iterations reached
```

## Extensibility Points

### 1. Custom Models
- Swap RLM_MODEL or SUBLM_MODEL in config
- Support for any OpenAI model
- Can extend to support other providers

### 2. Context Sources
Current: In-memory string variable
Future options:
- Vector databases (Pinecone, Weaviate)
- Document stores (Elasticsearch)
- SQL databases
- File systems
- APIs

### 3. Evaluation Criteria
Current: RLM decides when done
Future options:
- Confidence scoring
- Answer validation
- Fact checking
- Quality metrics

### 4. Caching Layer
Future addition:
- Cache common questions
- Store SubLLM responses
- Reduce API calls
- Improve latency

## Performance Characteristics

### Token Usage
- **RLM (Boss):** Higher tokens per request (GPT-4)
- **SubLLM (Intern):** Lower tokens (GPT-4o-mini)
- **Optimization:** Intern does heavy lifting with cheaper model

### Latency
- Total latency = (RLM latency + SubLLM latency) × iterations
- Typically 3-5 iterations for simple questions
- Complex questions may use all 5 iterations

### Cost Optimization
Using GPT-4o-mini for context reading (most expensive part) reduces costs significantly while maintaining quality.

## Testing Strategy

### Test Question Format
```json
{
  "id": "unique_id",
  "question": "User question",
  "context": "Knowledge base text",
  "expected_answer": "Expected result"
}
```

### Test Categories
1. **Simple Facts** - Direct lookup questions
2. **Multi-Step Reasoning** - Requires combining info
3. **Contextual Analysis** - Needs understanding relationships

## Logging and Debugging

### Console Output
- Color-coded by role (Boss/Intern)
- Shows each iteration clearly
- Final answer highlighted

### File Logs
- JSON format for easy parsing
- Full conversation history
- Timestamps and metadata
- Session-based organization

## Error Handling

### Configuration Errors
- Missing API key → Clear error message
- Invalid config → Validation with helpful messages

### API Errors
- Network issues → Graceful degradation
- Rate limits → Logged and reported
- Timeout → Retry logic (future)

### Logic Errors
- Max iterations reached → Synthesize best answer from accumulated info
- Max depth exceeded → Prevent infinite recursion
- Empty context → Early validation

## Security Considerations

### API Key Management
- Environment variables only
- Never committed to git
- .gitignore includes .env

### Input Validation
- Context size limits (future)
- Question sanitization (future)
- Rate limiting (future)

## Future Enhancements

1. **Vector Database Integration**
   - Semantic search in large contexts
   - Chunk management
   - Similarity matching

2. **Caching System**
   - Question similarity detection
   - Response reuse
   - Cache invalidation

3. **Metrics and Analytics**
   - Success rate tracking
   - Iteration analysis
   - Cost monitoring

4. **Web Interface**
   - REST API
   - WebSocket for real-time updates
   - Dashboard for monitoring

5. **Multi-Model Support**
   - Claude, Llama, etc.
   - Model comparison
   - Ensemble approaches

6. **Advanced Evaluation**
   - Answer quality scoring
   - Factual accuracy checking
   - Hallucination detection
