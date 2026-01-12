"""Improved RLM with better chunking and search strategies for REPL mode."""
from openai import OpenAI
from src.config import Config
from src.sublm import SubLLM
from src.logger import RLMLogger
from src.metrics import AnswerMetrics
from src.repl import SafeREPLEnvironment
from config.prompts import RLM_SYSTEM_PROMPT
import re


class RLMImproved:
    """
    Improved RLM with better strategies for finding scattered information.
    
    Key improvements:
    1. Smarter chunking (by section headers, not just paragraphs)
    2. Better keyword extraction from questions
    3. More systematic exploration of chunks
    4. Lower threshold for REPL mode (10K instead of 50K)
    """
    
    def __init__(self, model=None, max_iterations=None, max_depth=None, enable_metrics=True):
        """Initialize RLM."""
        Config.validate()
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = model or Config.RLM_MODEL
        self.max_iterations = max_iterations or Config.MAX_ITERATIONS
        self.max_depth = max_depth or Config.MAX_DEPTH
        self.sublm = SubLLM()
        self.logger = None
        self.enable_metrics = enable_metrics
        self.metrics_evaluator = AnswerMetrics() if enable_metrics else None
    
    def answer(self, user_question, context, depth=0):
        """
        Answer a question using appropriate strategy.
        
        Uses REPL mode for contexts > 10K characters (lower threshold for better handling).
        """
        # Initialize logger
        self.logger = RLMLogger()
        self.logger.log_start(user_question, context)
        
        # Check context size
        context_size = len(context)
        
        # Lower threshold: 10K instead of 50K
        split_threshold = getattr(Config, 'SPLIT_THRESHOLD', 10000)
        if context_size > split_threshold:
            self.logger.log_info(
                f"Large context detected ({context_size:,} chars > {split_threshold:,} threshold). Using REPL mode."
            )
            return self._answer_with_repl(user_question, context, depth)
        else:
            # Normal mode for smaller contexts
            return self._answer_normal(user_question, context, depth)
    
    def _smart_chunk_context(self, context, target_size=2500, max_size=5000, min_size=500):
        """
        Intelligently chunk context with size constraints.
        
        Improvements:
        1. Splits by section headers first
        2. Further splits oversized sections by paragraphs
        3. Maintains reasonable chunk sizes (target: 2500 chars)
        4. Never creates chunks < 500 or > 5000 characters
        
        This prevents:
        - 1 giant chunk (14K → 1 chunk) 
        - Tiny fragmented chunks (100 chars each)
        """
        chunks = []
        
        # Strategy 1: Split by section markers (>>>, ===, ---, ###, etc.)
        section_pattern = r'((?:>>>|===|---|###)\s*.+?(?:>>>|===|---|###))'
        sections = re.split(section_pattern, context)
        
        current_section = None
        current_content = []
        
        for section in sections:
            if not section.strip():
                continue
                
            # Check if this is a section header
            if re.match(r'^(>>>|===|---|###)', section.strip()):
                # Save previous section
                if current_section and current_content:
                    full_section = f"{current_section}\n\n" + "\n\n".join(current_content)
                    chunks.append(full_section)
                
                # Start new section
                current_section = section.strip()
                current_content = []
            else:
                # This is content - split it into paragraphs
                paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
                current_content.extend(paragraphs)
        
        # Don't forget the last section
        if current_section and current_content:
            full_section = f"{current_section}\n\n" + "\n\n".join(current_content)
            chunks.append(full_section)
        
        # If no sections found, fall back to paragraph splitting
        if not chunks:
            chunks = [p.strip() for p in context.split('\n\n') if p.strip()]
        
        # NEW: Handle oversized chunks by further splitting
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_size:
                # This chunk is too large - split by paragraphs
                paragraphs = chunk.split('\n\n')
                sub_chunk = ""
                
                for para in paragraphs:
                    # Would adding this paragraph exceed target size?
                    if len(sub_chunk) + len(para) + 2 > target_size and len(sub_chunk) >= min_size:
                        # Save current sub_chunk and start new one
                        final_chunks.append(sub_chunk)
                        sub_chunk = para
                    else:
                        # Add to current sub_chunk
                        sub_chunk = sub_chunk + "\n\n" + para if sub_chunk else para
                
                # Don't forget the last sub_chunk
                if sub_chunk and len(sub_chunk) >= min_size:
                    final_chunks.append(sub_chunk)
                elif sub_chunk and final_chunks:
                    # Too small - merge with previous
                    final_chunks[-1] += "\n\n" + sub_chunk
                elif sub_chunk:
                    # First chunk and too small - keep it anyway
                    final_chunks.append(sub_chunk)
            
            elif len(chunk) >= min_size:
                # Good size - keep as is
                final_chunks.append(chunk)
            
            elif final_chunks:
                # Too small - merge with previous chunk
                final_chunks[-1] += "\n\n" + chunk
            
            else:
                # First chunk and too small - keep it anyway
                final_chunks.append(chunk)
        
        return final_chunks if final_chunks else chunks
    
    def _extract_search_keywords(self, question):
        """
        Extract keywords from the question to help search for relevant chunks.
        
        Examples:
        - "What is the total revenue from all product lines in Q3 2024?"
          → ["revenue", "product lines", "Q3 2024", "Q3", "2024", "total"]
        - "What was the adjusted net profit?"
          → ["adjusted", "net profit", "profit"]
        """
        keywords = []
        
        # Extract quoted phrases first
        quoted = re.findall(r'"([^"]+)"', question)
        keywords.extend(quoted)
        
        # Extract key financial terms
        financial_terms = [
            "revenue", "profit", "income", "earnings", "sales", "cost", "expense",
            "margin", "growth", "loss", "EBITDA", "adjusted", "operating", "net",
            "gross", "cash flow", "product line", "Q1", "Q2", "Q3", "Q4",
            "quarter", "fiscal", "FY", "year", "2024", "2023", "total"
        ]
        
        question_lower = question.lower()
        for term in financial_terms:
            if term.lower() in question_lower:
                keywords.append(term)
        
        # Extract numbers and years
        numbers = re.findall(r'\\b\\d{4}\\b', question)  # Years like 2024
        keywords.extend(numbers)
        
        # Remove duplicates, keep order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def _answer_normal(self, user_question, context, depth=0):
        """Normal mode for smaller contexts (copied from original, works fine)."""
        if depth > self.max_depth:
            return f"Error: Max recursion depth ({self.max_depth}) reached"
        
        accumulated_info = []
        
        for iteration in range(1, self.max_iterations + 1):
            # Boss decides what to ask
            boss_question = self._formulate_question(
                user_question,
                accumulated_info,
                iteration
            )
            
            # Check if done
            if boss_question.startswith("FINAL_ANSWER:"):
                final_answer = boss_question.replace("FINAL_ANSWER:", "").strip()
                
                # Evaluate metrics BEFORE accepting
                final_metrics = None
                if self.enable_metrics:
                    final_metrics = self.metrics_evaluator.evaluate_answer(
                        user_question, final_answer, context, is_final=True
                    )
                    
                    # CRITICAL: Check if answer quality is acceptable
                    min_accuracy = getattr(Config, 'MIN_ACCURACY_THRESHOLD', 0.7)
                    accuracy = final_metrics.get('accuracy', 0)
                    overall = final_metrics.get('overall_score', 0)
                    
                    if accuracy < min_accuracy or overall < Config.EARLY_STOP_THRESHOLD:
                        # Boss wants to stop, but answer quality is too low - REJECT
                        self.logger.log_info(
                            f"⚠️  Boss tried to give final answer, but quality too low "
                            f"(accuracy={accuracy:.2f} < {min_accuracy:.2f} OR "
                            f"overall={overall:.2f} < {Config.EARLY_STOP_THRESHOLD:.2f}). "
                            f"Forcing boss to reconsider."
                        )
                        
                        # Log the rejected answer
                        self.logger.log_iteration(
                            iteration, "REJECTED_FINAL_ANSWER", 
                            f"Boss tried: {final_answer}", 
                            f"Rejected - accuracy too low",
                            {"metrics": final_metrics}
                        )
                        
                        # Add feedback to accumulated info so boss knows it was wrong
                        accumulated_info.append(
                            f"PREVIOUS ATTEMPT REJECTED: Your answer '{final_answer}' had "
                            f"accuracy={accuracy:.2f} (need ≥{min_accuracy:.2f}). "
                            f"Please reconsider by carefully comparing all the numbers."
                        )
                        
                        # Continue to next iteration instead of returning
                        continue
                
                self.logger.log_final_answer(final_answer, {
                    "total_iterations": iteration,
                    "metrics": final_metrics,
                    "mode": "normal"
                })
                return final_answer
            
            # Ask intern
            intern_answer = self.sublm.answer(boss_question, context)
            
            # Log iteration
            self.logger.log_iteration(
                iteration, "RLM_TO_SUBLM", boss_question, intern_answer, {"depth": depth}
            )
            
            # Generate intermediate answer
            intermediate_answer = self._generate_intermediate_answer(
                user_question,
                accumulated_info + [f"Q: {boss_question}\\nA: {intern_answer}"]
            )
            
            # Evaluate intermediate
            if self.enable_metrics:
                intermediate_metrics = self.metrics_evaluator.evaluate_answer(
                    user_question, intermediate_answer, context, is_final=False
                )
                
                self.logger.log_iteration(
                    iteration, "INTERMEDIATE_ANSWER", user_question, intermediate_answer,
                    {"iteration": iteration, "metrics": intermediate_metrics}
                )
                
                # Check for early stopping
                threshold = Config.EARLY_STOP_THRESHOLD
                min_accuracy = getattr(Config, 'MIN_ACCURACY_THRESHOLD', 0.7)
                overall_score = intermediate_metrics.get("overall_score", 0)
                accuracy_score = intermediate_metrics.get("accuracy", 0)
                
                # CRITICAL: Don't stop if accuracy is too low, even if overall score is good
                if accuracy_score < min_accuracy:
                    self.logger.log_info(
                        f"⚠️  Low accuracy detected ({accuracy_score:.2f} < {min_accuracy:.2f}). Forcing boss to reconsider."
                    )
                    # Don't add this to accumulated_info yet - let boss reconsider
                    # On next iteration, boss will see the low-accuracy answer and can correct it
                elif threshold > 0 and overall_score >= threshold:
                    self.logger.log_info(
                        f"✓ Early stopping: Quality threshold reached (overall={overall_score:.2f} >= {threshold:.2f}, accuracy={accuracy_score:.2f})"
                    )
                    
                    # Use intermediate answer as final
                    final_metrics = self.metrics_evaluator.evaluate_answer(
                        user_question, intermediate_answer, context, is_final=True
                    )
                    
                    self.logger.log_final_answer(intermediate_answer, {
                        "total_iterations": iteration,
                        "metrics": final_metrics,
                        "early_stopped": True,
                        "mode": "normal"
                    })
                    return intermediate_answer
            
            accumulated_info.append(f"Q: {boss_question}\\nA: {intern_answer}")
        
        # Max iterations reached
        final_answer = self._synthesize_final_answer(user_question, accumulated_info)
        
        final_metrics = None
        if self.enable_metrics:
            final_metrics = self.metrics_evaluator.evaluate_answer(
                user_question, final_answer, context, is_final=True
            )
        
        self.logger.log_final_answer(final_answer, {
            "total_iterations": self.max_iterations,
            "metrics": final_metrics,
            "reason": "max_iterations_reached"
        })
        
        return final_answer
    
    def _answer_with_repl(self, user_question, context, depth=0):
        """
        IMPROVED REPL mode with:
        1. Chunk-by-chunk evaluation and early stopping
        2. Context awareness of previous findings
        """
        # Initialize REPL
        repl = SafeREPLEnvironment()
        
        # Store context as variable (NOT in prompts!)
        repl.add_variable('context', context)
        repl.add_variable('user_question', user_question)
        
        # Add Sub_RLM function
        def Sub_RLM(query, text):
            """Boss can call this to query text chunks."""
            return self.sublm.answer(query, text)
        
        repl.add_function('Sub_RLM', Sub_RLM)
        
        # Track findings, explored chunks, and best answer
        findings = []  # Raw findings from each chunk
        explored_chunks = []  # Track which chunks we've looked at
        best_answer = None
        best_score = 0.0
        
        # IMPROVED: Use smart chunking
        chunks = self._smart_chunk_context(context)
        repl.add_variable('chunks', chunks)
        
        # IMPROVED: Extract keywords from question
        keywords = self._extract_search_keywords(user_question)
        repl.add_variable('keywords', keywords)
        
        # NEW: Make findings accessible to boss in REPL
        repl.add_variable('findings_so_far', findings)
        repl.add_variable('explored_chunks_indices', explored_chunks)
        
        self.logger.log_info(f"Split context into {len(chunks)} chunks for systematic exploration.")
        self.logger.log_info(f"Extracted search keywords: {keywords}")
        
        # REPL loop - guided exploration
        for iteration in range(1, self.max_iterations + 1):
            # NEW: Tell boss about previous findings and what's been explored
            code_or_answer = self._formulate_chunk_aware_code(
                user_question,
                findings,
                explored_chunks,
                iteration,
                len(chunks),
                keywords
            )
            
            # Check if boss says we're done
            if code_or_answer.startswith("FINAL:"):
                final_answer = code_or_answer.replace("FINAL:", "").strip()
                
                final_metrics = None
                if self.enable_metrics:
                    final_metrics = self.metrics_evaluator.evaluate_answer(
                        user_question, final_answer, context, is_final=True
                    )
                
                self.logger.log_final_answer(final_answer, {
                    "total_iterations": iteration,
                    "metrics": final_metrics,
                    "chunks_explored": len(explored_chunks),
                    "total_chunks": len(chunks),
                    "mode": "repl"
                })
                return final_answer
            
            # Execute the code
            self.logger.log_iteration(iteration, "BOSS_TO_REPL", "Code generated", code_or_answer, {})
            
            result = repl.execute(code_or_answer)
            
            if result["success"]:
                output = result["output"]
                self.logger.log_iteration(iteration, "REPL_OUTPUT", "Execution result", output, {})
                
                # Update explored_chunks from REPL namespace
                explored_chunks = repl.get_variable('explored_chunks_indices') or []
                
                # If we got output, process it
                if output and output.strip():
                    # Extract which chunks were examined (track this)
                    new_findings = output.strip()
                    findings.append(new_findings)
                    
                    # Update explored chunks list in REPL
                    repl.add_variable('findings_so_far', findings)
                    
                    # NEW: Generate intermediate answer AFTER EACH CHUNK
                    intermediate = self._synthesize_findings(user_question, findings)
                    
                    # NEW: Evaluate quality AFTER EACH CHUNK
                    if self.enable_metrics:
                        intermediate_metrics = self.metrics_evaluator.evaluate_answer(
                            user_question, intermediate, context, is_final=False
                        )
                        
                        overall_score = intermediate_metrics.get("overall_score", 0)
                        
                        self.logger.log_iteration(
                            iteration, "INTERMEDIATE_ANSWER", user_question, intermediate,
                            {"iteration": iteration, "metrics": intermediate_metrics, 
                             "chunks_explored": len(explored_chunks), "total_chunks": len(chunks)}
                        )
                        
                        self.logger.log_info(
                            f"Progress: Found {len(findings)} pieces | Explored {len(explored_chunks)}/{len(chunks)} chunks | Quality: {overall_score:.2f}"
                        )
                        
                        # Track best answer (use >= so ties update to latest)
                        if overall_score >= best_score:
                            best_score = overall_score
                            best_answer = intermediate
                            
                        # NOTE: No early stopping based on quality
                        # But stop if ALL chunks have been explored
                        if len(explored_chunks) >= len(chunks):
                            self.logger.log_info(f"All chunks explored ({len(explored_chunks)}/{len(chunks)}). Completing search.")
                            
                            # IMPORTANT: Use the LATEST intermediate answer (most complete)
                            # Not necessarily the "best" one (they might have same score)
                            final_answer_to_use = intermediate  # Always use latest after exploring all chunks
                            
                            final_metrics = self.metrics_evaluator.evaluate_answer(
                                user_question, final_answer_to_use, context, is_final=True
                            )
                            
                            self.logger.log_final_answer(final_answer_to_use, {
                                "total_iterations": iteration,
                                "metrics": final_metrics,
                                "reason": "all_chunks_explored",
                                "chunks_explored": len(explored_chunks),
                                "total_chunks": len(chunks),
                                "best_score_achieved": best_score,
                                "mode": "repl"
                            })
                            return final_answer_to_use
            else:
                error_msg = result["error"]
                self.logger.log_iteration(iteration, "REPL_ERROR", "Execution failed", error_msg, {})
                self.logger.log_info(f"Code execution failed: {error_msg}")
        
        # All iterations complete - return best answer found
        if best_answer:
            self.logger.log_info(f"Explored all {len(explored_chunks)}/{len(chunks)} chunks. Using best answer found (quality: {best_score:.2f})")
            final_answer = best_answer
        else:
            self.logger.log_info(f"Explored all {len(explored_chunks)}/{len(chunks)} chunks. Synthesizing final answer.")
            final_answer = self._synthesize_findings(user_question, findings)
        
        final_metrics = None
        if self.enable_metrics:
            final_metrics = self.metrics_evaluator.evaluate_answer(
                user_question, final_answer, context, is_final=True
            )
        
        self.logger.log_final_answer(final_answer, {
            "total_iterations": iteration if iteration <= self.max_iterations else self.max_iterations,
            "metrics": final_metrics,
            "reason": "explored_all_chunks",
            "chunks_explored": len(explored_chunks),
            "total_chunks": len(chunks),
            "best_score_achieved": best_score,
            "mode": "repl"
        })
        
        return final_answer
    
    def _formulate_chunk_aware_code(self, user_question, findings, explored_chunks, iteration, total_chunks, keywords):
        """
        NEW: Boss formulates code with awareness of:
        1. What we've found so far
        2. Which chunks we've already explored
        3. Current quality of answer
        """
        # Show recent findings (last 3)
        findings_summary = "\\n".join(findings[-3:]) if findings else "No findings yet."
        
        # Create keyword hints
        keyword_hints = ", ".join(f"'{kw}'" for kw in keywords[:5])
        
        # Calculate progress
        chunks_explored = len(explored_chunks)
        chunks_remaining = total_chunks - chunks_explored
        
        prompt = f"""You're solving: "{user_question}"

ENVIRONMENT:
- Variable 'chunks': Context split into {total_chunks} chunks (by sections/headers)
- Variable 'keywords': {keywords}
- Variable 'findings_so_far': {len(findings)} findings from previous chunks
- Variable 'explored_chunks_indices': Which chunks you've already looked at
- Function 'Sub_RLM(query, text)': Ask intern to analyze text

PROGRESS:
- Chunks explored: {chunks_explored}/{total_chunks}
- Chunks remaining: {chunks_remaining}
- Findings so far: {len(findings)} pieces

Recent findings:
{findings_summary}

Iteration {iteration}/{self.max_iterations}

YOUR STRATEGY:
1. **Build answer progressively**: As you explore each chunk, accumulate findings
   - Findings so far are in: findings_so_far
   - Keep adding to it as you find more information

2. **Search systematically**: 
   - Use keywords: {keyword_hints}
   - Find relevant chunks you HAVEN'T explored yet
   - Example: relevant = [i for i, c in enumerate(chunks) if i not in explored_chunks_indices and any(kw.lower() in c.lower() for kw in keywords)]

3. **Track your progress**:
   - Mark chunks as explored: explored_chunks_indices.append(idx)
   - Add findings: findings_so_far.append(result)
   - Print your findings so they're saved

4. **Keep going until all chunks explored**:
   - Don't stop early - we want to check all chunks
   - But if you've found complete answer and no more relevant chunks, you can use FINAL:
   - Otherwise, keep exploring until you've checked all relevant chunks

EXAMPLE CODE:
# Step 1: Find NEW chunks to explore (not already explored)
unexplored = [i for i, c in enumerate(chunks) if i not in explored_chunks_indices]
relevant = [i for i in unexplored if any(kw.lower() in chunks[i].lower() for kw in keywords)]

# Step 2: Explore NEW chunks systematically
if relevant:
    for idx in relevant[:3]:  # Check up to 3 new chunks per iteration
        explored_chunks_indices.append(idx)  # Mark as explored
        result = Sub_RLM("{user_question}", chunks[idx])
        print(f"Chunk {{idx}}: {{result}}")
        findings_so_far.append(result)  # Add to findings
else:
    # No more relevant chunks
    print("All relevant chunks explored")

Write Python code (NO 'FINAL:' or 'FINAL_ANSWER:' - those cause syntax errors!):"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RLM_SYSTEM_PROMPT + "\\n\\nYou are chunk-aware. Check findings before exploring more chunks. Stop early if you have the answer!"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=600
            )
            
            result = response.choices[0].message.content.strip()
            
            # Remove markdown fences
            if result.startswith("```python"):
                result = result[9:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            
            return result.strip()
        except Exception as e:
            return f"FINAL: Error: {str(e)}"
    
    def _formulate_guided_code_improved(self, user_question, findings, iteration, total_chunks, keywords):
        """IMPROVED: Boss formulates better code with keyword hints."""
        findings_summary = "\\n".join(findings[-3:]) if findings else "No findings yet."
        
        # Create keyword hints
        keyword_hints = ", ".join(f"'{kw}'" for kw in keywords[:5])  # Show first 5 keywords
        
        prompt = f"""You're solving: "{user_question}"

ENVIRONMENT:
- Variable 'chunks': Context split into {total_chunks} chunks (by sections/headers)
- Variable 'keywords': {keywords} (extracted from question)
- Function 'Sub_RLM(query, text)': Ask intern to analyze text

Progress: Found {len(findings)} pieces of information
Recent findings:
{findings_summary}

Iteration {iteration}/{self.max_iterations}

YOUR TASK - Be SYSTEMATIC:
1. Search for relevant chunks using multiple keywords
   Example: relevant = [c for c in chunks if any(kw.lower() in c.lower() for kw in keywords)]
2. If nothing found, try broader search terms
3. Ask intern specific questions about each chunk
4. ALWAYS print your findings: print(f"Found: {{result}}")

SEARCH STRATEGY:
- For financial questions, look for: {keyword_hints}
- Try section headers like "CFO", "FINANCIAL", "REVENUE", "PRODUCT LINE"
- Check for table-like content, numbered lists, bullet points

Example code:
# Cast a wide net first
relevant = [c for c in chunks if any(kw.lower() in c.lower() for kw in ['revenue', 'q3', '2024', 'product'])]
if relevant:
    for chunk in relevant[:3]:  # Check first 3 matches
        result = Sub_RLM("{user_question}", chunk)
        print(f"Found in chunk: {{result}}")

When you have the complete answer, use:
FINAL: Your complete answer here

Write Python code:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RLM_SYSTEM_PROMPT + "\\n\\nYou systematically search chunks using keywords. ALWAYS print findings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            
            # Remove markdown fences
            if result.startswith("```python"):
                result = result[9:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            
            return result.strip()
        except Exception as e:
            return f"FINAL: Error: {str(e)}"
    
    # Copy other helper methods from original RLM...
    def _is_ranking_question(self, question):
        """
        Detect if a question involves rankings or comparisons.
        These questions need special attention to preserve qualifiers.
        """
        ranking_terms = [
            'largest', 'smallest', 'biggest', 'tiniest',
            'highest', 'lowest', 'tallest', 'shortest',
            'first', 'second', 'third', 'fourth', 'fifth',
            'top', 'bottom', 'best', 'worst',
            'most', 'least', 'maximum', 'minimum',
            'greater', 'lesser', 'superior', 'inferior',
            'leading', 'trailing', 'primary', 'secondary'
        ]
        question_lower = question.lower()
        return any(term in question_lower for term in ranking_terms)
    
    def _formulate_question(self, user_question, accumulated_info, iteration):
        """Boss formulates next question (from original)."""
        context_summary = "\\n\\n".join(accumulated_info) if accumulated_info else "No information gathered yet."
        
        prompt = f"""Original question: {user_question}

Information gathered so far:
{context_summary}

This is iteration {iteration}. What specific question should I ask next to make progress?

If you have enough information to answer the original question, respond with:
FINAL_ANSWER: [your complete answer]

Otherwise, formulate a specific, focused question."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RLM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            boss_question = response.choices[0].message.content.strip()
            
            # PHASE 1 ENHANCEMENT: If original question involves rankings, 
            # instruct intern to include qualifiers
            if self._is_ranking_question(user_question) and not boss_question.startswith("FINAL_ANSWER:"):
                boss_question += " (Include any contextual qualifiers like 'globally', 'regionally', 'worldwide', 'in the industry', or descriptive phrases that add important context.)"
            
            return boss_question
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _synthesize_findings(self, user_question, findings):
        """Synthesize answer from accumulated findings - IMPROVED to extract and aggregate data."""
        if not findings:
            return "No information found yet."
        
        findings_text = "\\n".join(findings)
        
        prompt = f"""Question: {user_question}

Findings from exploring the document:
{findings_text}

IMPORTANT INSTRUCTIONS:
1. Look for SPECIFIC DATA in the findings (numbers, names, amounts, etc.)
2. If findings say "not provided" or "not mentioned", IGNORE that finding and look at others
3. If you find multiple pieces of data (e.g., revenues for different products), COMBINE them
4. Calculate totals if needed (e.g., sum up all revenues)
5. Provide a COMPLETE answer with all the data you found

Examples:
- Finding 1: "Cloud: $87.3M" 
  Finding 2: "Security: $56.8M"
  → Answer: "Total revenue: $144.1M (Cloud: $87.3M + Security: $56.8M)"

- Finding 1: "not provided"
  Finding 2: "Employee count: 450"
  → Answer: "450 employees" (ignore finding 1)

Synthesize a clear, complete answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract and aggregate data from findings. Ignore findings that say 'not provided'. Calculate totals when appropriate."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more accurate data extraction
                max_tokens=400
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return findings_text
    
    def _synthesize_final_answer(self, original_question, accumulated_info):
        """Synthesize answer from gathered info."""
        context_summary = "\\n\\n".join(accumulated_info)
        
        prompt = f"""Question: {original_question}

Information gathered:
{context_summary}

Provide the best answer based on this information."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Synthesize clear answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _generate_intermediate_answer(self, original_question, accumulated_info):
        """Generate intermediate answer."""
        context_summary = "\\n\\n".join(accumulated_info)
        
        prompt = f"""Question: {original_question}

Info so far:
{context_summary}

Provide preliminary answer. Start with: "Based on what I've gathered so far: ..." """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Provide intermediate answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_config_info(self):
        """Return config info."""
        return {
            "rlm_model": self.model,
            "sublm_model": self.sublm.get_model_name(),
            "max_iterations": self.max_iterations,
            "max_depth": self.max_depth
        }
# Alias for backward compatibility
RLM = RLMImproved