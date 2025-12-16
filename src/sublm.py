"""SubLLM (Intern) - Reads context and answers specific questions."""
from openai import OpenAI
from src.config import Config
from config.prompts import SUBLM_SYSTEM_PROMPT


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
        # Construct the prompt with context
        user_message = f"""Context:
{context}

Question from boss: {question}

Please answer the question based solely on the context provided above. If the answer is not in the context, clearly state that."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SUBLM_SYSTEM_PROMPT},
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
