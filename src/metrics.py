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
        accuracy = self._evaluate_accuracy(answer, context)
        
        # Get second opinion if accuracy seems suspiciously low
        accuracy = self._verify_with_second_opinion(answer, context, accuracy)
        metrics['accuracy'] = accuracy
        
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
        prompt = f"""Evaluate if this answer COMPLETELY addresses the question.

Question: {question}
Answer: {answer}

CRITICAL RULES FOR "ALL" and "TOTAL" QUESTIONS:
- If question asks for "ALL" or "TOTAL", the answer MUST include ALL items
- Score 1.0 ONLY if answer explicitly confirms completeness (e.g., "all 6 products", "complete list", "every office")
- Partial lists get partial scores based on coverage
- Without explicit completeness confirmation, max score is 0.9

EXAMPLES FOR "ALL/TOTAL" QUESTIONS:
- Q: "Total from ALL product lines?" A: "$144M (Cloud + Security)" -> 0.4 (only 2, clearly partial)
- Q: "Total from ALL product lines?" A: "$257M (5 products)" -> 0.85 (5 products but no "all" confirmation)
- Q: "Total from ALL product lines?" A: "$306M (all 6 products: Cloud, Security, Analytics, Developer, AI/ML, Collaboration)" -> 1.0 (explicitly complete)
- Q: "Total employees across ALL offices?" A: "1450 (7 offices)" -> 0.9 (comprehensive but no explicit "all")
- Q: "Total employees across ALL offices?" A: "1450 from all 7 offices: Seattle, London..." -> 1.0 (explicit "all")

EXAMPLES FOR SIMPLE QUESTIONS:
- Q: "How much revenue growth?" A: "$1.2M" -> 1.0 (complete)
- Q: "How much revenue growth?" A: "Revenue grew" -> 0.3 (no number)
- Q: "Who founded the company?" A: "Jane and Robert" -> 1.0 (complete)

SCORING FOR "ALL/TOTAL" QUESTIONS:
- 1.0 = ALL items with explicit confirmation ("all 6", "every", "complete")
- 0.9 = Comprehensive list but no explicit "all" (e.g., "6 products" without "all 6")
- 0.85 = Nearly all (5/6 items)
- 0.7 = Most (4/6 items)
- 0.5 = Some (2-3/6 items)  
- 0.3-0.4 = Few (1-2/6 items)
- 0.0 = None

SCORING FOR SIMPLE QUESTIONS:
- 1.0 = Directly provides complete answer
- 0.7 = Mostly complete
- 0.5 = Partial
- 0.3 = Barely addresses
- 0.0 = Doesn't answer

CRITICAL: 
- For "all/total" questions, 1.0 requires explicit completeness language
- 5 out of 6 items → max 0.85 (unless confirmed complete)
- "all", "every", "complete", "entire" → indicators of completeness

Respond ONLY with a number between 0.0 and 1.0. No explanation."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an answer completeness evaluator. A complete answer directly provides what was asked, even if brief. Rambling without answering scores low. Respond only with a number."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            
            # Sanity check
            if score < 0.0 or score > 1.0:
                print(f"Warning: Invalid completeness score {score}, defaulting to 0.5")
                return 0.5
            
            return score
            
        except Exception as e:
            print(f"Error evaluating completeness: {e}")
            return 0.5  # Default middle score
    
    def _evaluate_accuracy(self, answer, context):
        """
        Evaluate if the answer is factually correct based on context.
        
        Returns:
            float: Score from 0.0 to 1.0
        """
        prompt = f"""You are a fact checker. Evaluate if this answer is factually accurate based on the context.

Context (source of truth):
{context}

Answer to evaluate:
{answer}

Instructions:
1. Check if the facts in the answer match the context
2. Ignore formatting or wording differences - focus on factual correctness
3. If the answer is mathematically derived from context data, verify the calculation

Examples:
- Context says "Revenue Q1: $2.3M, Q3: $3.5M", Answer says "Growth was $1.2M" → 1.0 (correct math: 3.5-2.3=1.2)
- Context says "Founded in 2010", Answer says "Founded in 2012" → 0.0 (wrong year)
- Context says "Founded by Jane and Robert", Answer says "Founded by Jane Smith" → 0.7 (partially correct)

Score:
- 1.0 = Completely accurate (all facts correct)
- 0.9 = Nearly accurate (very minor issues)
- 0.7 = Mostly accurate (small errors or incomplete)
- 0.5 = Half accurate (mix of right and wrong)
- 0.3 = Mostly inaccurate (mostly wrong)
- 0.0 = Completely wrong (all facts incorrect)

Respond ONLY with a number between 0.0 and 1.0. No explanation."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise fact checker. Verify mathematical calculations and factual accuracy. Respond only with a number."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            
            # Sanity check: if score is outside valid range, default to 0.5
            if score < 0.0 or score > 1.0:
                print(f"Warning: Invalid accuracy score {score}, defaulting to 0.5")
                return 0.5
            
            return score
            
        except Exception as e:
            print(f"Error evaluating accuracy: {e}")
            return 0.5
    
    def _verify_with_second_opinion(self, answer, context, first_score):
        """
        Get a second opinion on accuracy if first score seems wrong.
        
        Args:
            answer: The answer to verify
            context: The context
            first_score: The initial accuracy score
            
        Returns:
            float: Verified score
        """
        # If first score is suspiciously low (0.0-0.2) for what looks like a reasonable answer,
        # get a second opinion
        if first_score < 0.3 and len(answer) > 10:  # Non-trivial answer
            try:
                prompt = f"""Double-check this accuracy evaluation.

Context: {context}
Answer: {answer}
Previous score: {first_score}

Is this answer actually incorrect, or was it scored too harshly?
Give a fair accuracy score from 0.0 to 1.0.

Respond ONLY with a number."""

                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are reviewing an accuracy score. Be fair and precise."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=10
                )
                
                second_score = float(response.choices[0].message.content.strip())
                
                # If scores differ significantly, take the higher one (benefit of doubt)
                if abs(second_score - first_score) > 0.3:
                    print(f"Accuracy review: First={first_score}, Second={second_score}, Using={second_score}")
                    return second_score
                    
            except Exception as e:
                print(f"Error in second opinion: {e}")
        
        return first_score
    
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