import os
from loguru import logger
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

class GroqClient:
    """Wrapper around Groq API for LLM operations"""
    
    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL):
        self.api_key = api_key
        self.model = model
        self.client = Groq(api_key=api_key)
        logger.info(f"GroqClient initialized with model: {model}")
    
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """
        Generate text using Groq API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-2.0)
        
        Returns:
            Generated text
        """
        try:
            message = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return message.choices[0].message.content
        except Exception as e:
            logger.exception("Groq API call failed")
            raise

# Global instance
groq_client = GroqClient()