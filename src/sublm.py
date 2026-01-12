"""SubLLM (Intern) - Reads context and answers specific questions."""
from openai import OpenAI
from src.config import Config

# Try to import from config, fallback to inline if not available
try:
    from config.prompts import SUBLM_SYSTEM_PROMPT
    SUBLM_SYSTEM_PROMPT_BASE = SUBLM_SYSTEM_PROMPT
except ImportError:
    SUBLM_SYSTEM_PROMPT_BASE = "You are a helpful assistant that answers questions based on provided context."


class SubLLM:
    """Sub Language Model - the intern that reads context."""
    
    def __init__(self, model=None):
        """Initialize SubLLM with optional model override."""
        Config.validate()
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = model or Config.SUBLM_MODEL
    
    def answer(self, question, context):
        """
        Answer a specific question based on the provided context.
        
        Args:
            question: The question from the boss (RLM)
            context: The knowledge base/context to read from
            
        Returns:
            str: Answer to the question based on context
        """
        # Enhanced system prompt for better context preservation
        enhanced_system_prompt = f"""{SUBLM_SYSTEM_PROMPT_BASE}

IMPORTANT - When answering questions about rankings, sizes, or comparisons:
1. Include the actual numbers/data
2. Include contextual qualifiers that add meaning (e.g., "globally", "regionally", "worldwide", "in the industry", "excluding X")
3. Quote relevant phrases that provide important context
4. Preserve descriptive language from the original text

Examples:
❌ Bad: "Team A has 100 engineers"
✅ Good: "Team A has 100 engineers (described in the context as 'our second-largest engineering center globally')"

❌ Bad: "Revenue was $50M"
✅ Good: "Revenue was $50M (excluding one-time charges of $10M)"

❌ Bad: "Office B is larger"  
✅ Good: "Office B has 380 employees (described as 'second-largest globally'), Office A has 280 employees"

This helps preserve important context that might change the interpretation of the data."""

        # Construct the prompt with context
        user_message = f"""Context:
{context}

Question from boss: {question}

Please answer the question based solely on the context provided above. If the answer is not in the context, clearly state that."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,  # Lower temperature for more focused answers
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error answering question: {str(e)}"
    
    def get_model_name(self):
        """Return the model being used."""
        return self.model