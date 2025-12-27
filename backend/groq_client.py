# # backend/groq_client.py
# import os
# import httpx
# from loguru import logger

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
# GROQ_BASE = os.getenv("GROQ_BASE", "https://api.groq.com/openai/v1")

# class GroqClient:
#     def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL):
#         if not api_key:
#             raise ValueError("GROQ_API_KEY not set")
#         self.api_key = api_key
#         self.model = model
#         self.base = GROQ_BASE
#         self.client = httpx.Client(timeout=30.0)

#     def _headers(self):
#         return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

#     def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0):
#         """
#         Send prompt to Groq LLM for generation. Returns text.
#         Uses OpenAI-compatible chat completions API.
#         """
#         payload = {
#             "model": self.model,
#             "messages": [{"role": "user", "content": prompt}],
#             "max_tokens": max_tokens,
#             "temperature": temperature,
#         }
#         try:
#             resp = self.client.post(f"{self.base}/chat/completions", json=payload, headers=self._headers())
#             resp.raise_for_status()
#             data = resp.json()
#             text = data["choices"][0]["message"]["content"]
#             return text
#         except Exception as e:
#             logger.exception("Groq generate failed")
#             raise

# # convenience
# groq_client = GroqClient()

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