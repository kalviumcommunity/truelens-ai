# core/llm.py
import os
from groq import Groq

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(prompt: str, model: str = "llama3-70b-8192", temperature: float = 0.7) -> str:
    """
    Calls Groq LLM with the given prompt and returns the response text.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ LLM call failed: {str(e)}"