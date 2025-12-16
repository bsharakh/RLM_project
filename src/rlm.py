"""RLM (Root Language Model) - The boss that orchestrates questioning."""
from openai import OpenAI
from src.config import Config
from src.sublm import SubLLM
from src.logger import RLMLogger
from src.metrics import AnswerMetrics
from config.prompts import RLM_SYSTEM_PROMPT


class RLM:
    """Root Language Model - the boss that delegates to interns."""
    
    def __init__(self, model=None, max_iterations=None, max_depth=None, enable_metrics=True):
        """
        Initialize RLM.
        
        Args:
            model: Optional model override for RLM
            max_iterations: Max iterations per question (default from config)
            max_depth: Max recursion depth (default from config)
            enable_metrics: Whether to evaluate answer quality metrics
        """
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
        Answer a user question using recursive delegation to SubLLM.
        
        Args:
            user_question: The original question from the user
            context: The knowledge base/context to query
            depth: Current recursion depth
            
        Returns:
            str: Final answer to the user's question
        """
        # Initialize logger for this question
        self.logger = RLMLogger()
        self.logger.log_start(user_question, context)
        
        # Check max depth
        if depth > self.max_depth:
            error_msg = f"Max recursion depth ({self.max_depth}) reached"
            self.logger.log_error(error_msg)
            return f"Error: {error_msg}"
        
        # Start the iterative process
        conversation_history = []
        accumulated_info = []
        
        for iteration in range(1, self.max_iterations + 1):
            # Boss decides what to ask the intern
            boss_question = self._formulate_question(
                user_question,
                accumulated_info,
                iteration
            )
            
            if boss_question.startswith("FINAL_ANSWER:"):
                # Boss is satisfied, extract the final answer
                final_answer = boss_question.replace("FINAL_ANSWER:", "").strip()
                
                # Evaluate final answer metrics
                final_metrics = None
                if self.enable_metrics:
                    final_metrics = self.metrics_evaluator.evaluate_answer(
                        user_question,
                        final_answer,
                        context,
                        is_final=True
                    )
                
                self.logger.log_iteration(
                    iteration,
                    "FINAL_ANSWER",
                    user_question,
                    final_answer,
                    {
                        "total_iterations": iteration,
                        "metrics": final_metrics
                    }
                )
                return final_answer
            
            # Ask the intern (SubLLM)
            intern_answer = self.sublm.answer(boss_question, context)
            
            # Log this iteration
            self.logger.log_iteration(
                iteration,
                "RLM_TO_SUBLM",
                boss_question,
                intern_answer,
                {"depth": depth}
            )
            
            # Generate intermediate answer (what we know so far)
            intermediate_answer = self._generate_intermediate_answer(
                user_question,
                accumulated_info + [f"Q: {boss_question}\nA: {intern_answer}"]
            )
            
            # Evaluate intermediate answer metrics
            intermediate_metrics = None
            if self.enable_metrics:
                intermediate_metrics = self.metrics_evaluator.evaluate_answer(
                    user_question,
                    intermediate_answer,
                    context,
                    is_final=False
                )
            
            # Log intermediate answer with metrics
            self.logger.log_iteration(
                iteration,
                "INTERMEDIATE_ANSWER",
                user_question,
                intermediate_answer,
                {
                    "iteration": iteration,
                    "is_intermediate": True,
                    "metrics": intermediate_metrics
                }
            )
            
            # Add to conversation history
            conversation_history.append({
                "question": boss_question,
                "answer": intern_answer
            })
            accumulated_info.append(f"Q: {boss_question}\nA: {intern_answer}")
        
        # Max iterations reached, formulate best answer from what we have
        final_answer = self._synthesize_final_answer(user_question, accumulated_info)
        
        # Evaluate final answer metrics
        final_metrics = None
        if self.enable_metrics:
            final_metrics = self.metrics_evaluator.evaluate_answer(
                user_question,
                final_answer,
                context,
                is_final=True
            )
        
        self.logger.log_iteration(
            self.max_iterations,
            "FINAL_ANSWER",
            user_question,
            final_answer,
            {
                "reason": "max_iterations_reached",
                "metrics": final_metrics
            }
        )
        
        return final_answer
    
    def _formulate_question(self, original_question, accumulated_info, iteration):
        """
        Formulate the next question to ask the intern.
        
        Args:
            original_question: The user's original question
            accumulated_info: List of previous Q&A pairs
            iteration: Current iteration number
            
        Returns:
            str: Next question for the intern, or "FINAL_ANSWER: ..." if done
        """
        # Build context for the boss
        context_summary = "\n\n".join(accumulated_info) if accumulated_info else "No information gathered yet."
        
        prompt = f"""Original user question: {original_question}

Information gathered so far from intern:
{context_summary}

This is iteration {iteration}/{self.max_iterations}.

Based on the information gathered, either:
1. If you have enough information to answer the user's question, respond with: FINAL_ANSWER: <your complete answer>
2. If you need more information, formulate a specific question to ask your intern.

Remember: Your intern can only read the context and answer your questions. Be strategic."""
        
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
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.log_error(f"Error formulating question: {str(e)}")
            return f"FINAL_ANSWER: Error occurred while processing: {str(e)}"
    
    def _synthesize_final_answer(self, original_question, accumulated_info):
        """
        Synthesize a final answer from accumulated information.
        
        Args:
            original_question: The user's original question
            accumulated_info: List of all Q&A pairs
            
        Returns:
            str: Final synthesized answer
        """
        context_summary = "\n\n".join(accumulated_info)
        
        prompt = f"""Original user question: {original_question}

All information gathered from intern:
{context_summary}

Please provide the best possible answer to the user's question based on the information gathered."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that synthesizes information into clear answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error synthesizing answer: {str(e)}"
    
    def _generate_intermediate_answer(self, original_question, accumulated_info):
        """
        Generate an intermediate answer showing what we know so far.
        
        Args:
            original_question: The user's original question
            accumulated_info: List of Q&A pairs so far
            
        Returns:
            str: Intermediate answer with caveat that more thinking may be needed
        """
        context_summary = "\n\n".join(accumulated_info)
        
        prompt = f"""Original user question: {original_question}

Information gathered so far:
{context_summary}

Based on the information gathered SO FAR, provide a preliminary answer. 
Start with: "Based on what I've gathered so far: ..." 
Be honest if the information is incomplete."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant providing intermediate answers while gathering more information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating intermediate answer: {str(e)}"
    
    def get_config_info(self):
        """Return current configuration."""
        return {
            "rlm_model": self.model,
            "sublm_model": self.sublm.get_model_name(),
            "max_iterations": self.max_iterations,
            "max_depth": self.max_depth
        }