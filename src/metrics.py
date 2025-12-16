"""Metrics for evaluating intermediate and final answers."""
from openai import OpenAI
from src.config import Config


class AnswerMetrics:
    """Evaluate answer quality using various metrics."""
    
    def __init__(self):
        """Initialize metrics evaluator."""
        Config.validate()
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    def evaluate_answer(self, question, answer, context, is_final=False):
        """
        Evaluate an answer using multiple metrics.
        
        Args:
            question: The original user question
            answer: The answer to evaluate
            context: The ground truth context
            is_final: Whether this is the final answer or intermediate
            
        Returns:
            dict: Metrics including completeness, accuracy, confidence
        """
        metrics = {}
        
        # Completeness: Does it answer all parts of the question?
        metrics['completeness'] = self._evaluate_completeness(question, answer)
        
        # Accuracy: Is the answer factually correct based on context?
        metrics['accuracy'] = self._evaluate_accuracy(answer, context)
        
        # Confidence: How confident is the answer?
        metrics['confidence'] = self._evaluate_confidence(answer, question, context)
        
        # Relevance: How relevant is the answer to the question?
        metrics['relevance'] = self._evaluate_relevance(question, answer)
        
        # Overall score (weighted average)
        metrics['overall_score'] = self._calculate_overall_score(metrics, is_final)
        
        return metrics
    
    def _evaluate_completeness(self, question, answer):
        """
        Evaluate if the answer addresses all parts of the question.
        
        Returns:
            float: Score from 0.0 to 1.0
        """
        prompt = f"""Evaluate the completeness of this answer.

Question: {question}
Answer: {answer}

Does the answer address ALL parts of the question? 
- If it fully answers everything: score 1.0
- If it partially answers: score 0.3-0.7 based on how much
- If it doesn't answer at all: score 0.0

Respond ONLY with a number between 0.0 and 1.0. No explanation."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for metrics
                messages=[
                    {"role": "system", "content": "You are an answer evaluator. Respond only with a number."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            return float(score_text)
            
        except Exception as e:
            print(f"Error evaluating completeness: {e}")
            return 0.5  # Default middle score
    
    def _evaluate_accuracy(self, answer, context):
        """
        Evaluate if the answer is factually correct based on context.
        
        Returns:
            float: Score from 0.0 to 1.0
        """
        prompt = f"""Evaluate the factual accuracy of this answer against the context.

Context (ground truth):
{context}

Answer to evaluate:
{answer}

Is the answer factually accurate based on the context?
- If completely accurate: score 1.0
- If mostly accurate with minor errors: score 0.7-0.9
- If partially accurate: score 0.3-0.6
- If mostly inaccurate: score 0.1-0.3
- If completely wrong: score 0.0

Respond ONLY with a number between 0.0 and 1.0. No explanation."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a fact checker. Respond only with a number."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            return float(score_text)
            
        except Exception as e:
            print(f"Error evaluating accuracy: {e}")
            return 0.5
    
    def _evaluate_confidence(self, answer, question, context):
        """
        Evaluate the confidence level of the answer.
        Uses LLM to assess how well-supported the answer is by the context.
        
        Args:
            answer: The answer to evaluate
            question: The original question
            context: The source context
        
        Returns:
            float: Score from 0.0 to 1.0
        """
        prompt = f"""Evaluate how well-supported this answer is by the available information.

Question: {question}
Answer: {answer}
Context: {context}

Consider:
- Does the context provide clear evidence for this answer?
- Is the answer based on solid facts or speculation?
- Are there gaps in information that make the answer uncertain?

Rate the confidence/certainty of this answer:
- 1.0 = Highly confident, strong evidence in context
- 0.8 = Confident, good evidence
- 0.6 = Moderate confidence, some evidence
- 0.4 = Low confidence, weak evidence
- 0.2 = Very uncertain, minimal evidence
- 0.0 = Pure speculation, no evidence

Respond ONLY with a number between 0.0 and 1.0. No explanation."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a confidence evaluator. Respond only with a number."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            return float(score_text)
            
        except Exception as e:
            print(f"Error evaluating confidence: {e}")
            return 0.5
    
    def _evaluate_relevance(self, question, answer):
        """
        Evaluate if the answer is relevant to the question.
        
        Returns:
            float: Score from 0.0 to 1.0
        """
        prompt = f"""Evaluate the relevance of this answer to the question.

Question: {question}
Answer: {answer}

Is the answer relevant and on-topic?
- If highly relevant and directly addresses the question: score 1.0
- If mostly relevant: score 0.7-0.9
- If somewhat relevant: score 0.4-0.6
- If barely relevant: score 0.1-0.3
- If completely off-topic: score 0.0

Respond ONLY with a number between 0.0 and 1.0. No explanation."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a relevance evaluator. Respond only with a number."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            return float(score_text)
            
        except Exception as e:
            print(f"Error evaluating relevance: {e}")
            return 0.5
    
    def _calculate_overall_score(self, metrics, is_final):
        """
        Calculate weighted overall score.
        
        For intermediate answers: Confidence matters less
        For final answers: All metrics weighted equally
        
        Args:
            metrics: Dictionary of individual metric scores
            is_final: Whether this is a final answer
            
        Returns:
            float: Overall score from 0.0 to 1.0
        """
        if is_final:
            # Final answer: all metrics equal weight
            weights = {
                'completeness': 0.35,
                'accuracy': 0.35,
                'confidence': 0.15,
                'relevance': 0.15
            }
        else:
            # Intermediate answer: accuracy matters most, confidence matters less
            weights = {
                'completeness': 0.25,
                'accuracy': 0.45,
                'confidence': 0.05,
                'relevance': 0.25
            }
        
        overall = (
            metrics['completeness'] * weights['completeness'] +
            metrics['accuracy'] * weights['accuracy'] +
            metrics['confidence'] * weights['confidence'] +
            metrics['relevance'] * weights['relevance']
        )
        
        return round(overall, 3)
    
    def format_metrics(self, metrics):
        """
        Format metrics for display.
        
        Args:
            metrics: Dictionary of metric scores
            
        Returns:
            str: Formatted metrics string
        """
        return f"""Metrics:
  Completeness: {metrics['completeness']:.2f} (answers all parts of question)
  Accuracy:     {metrics['accuracy']:.2f} (factually correct)
  Confidence:   {metrics['confidence']:.2f} (certainty level)
  Relevance:    {metrics['relevance']:.2f} (on-topic)
  ─────────────────────────
  Overall:      {metrics['overall_score']:.2f}"""
    
    def get_score_interpretation(self, score):
        """
        Get human-readable interpretation of a score.
        
        Args:
            score: Numeric score from 0.0 to 1.0
            
        Returns:
            str: Interpretation (Excellent, Good, Fair, Poor)
        """
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.75:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        elif score >= 0.4:
            return "Poor"
        else:
            return "Very Poor"