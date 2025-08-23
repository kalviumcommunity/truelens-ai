from core import llm

def build_dynamic_prompt(user_input: str, context: str = None, style: str = "balanced") -> str:
    """
    Dynamically builds a prompt based on user input, context, and style.
    
    Args:
        user_input (str): Raw user query
        context (str, optional): Retrieved knowledge or embeddings context
        style (str, optional): Prompting style ("formal", "casual", "balanced")

    Returns:
        str: Final constructed prompt
    """

    base_instruction = "You are TrueLens-AI, an authenticity checker. Always provide reliable, verifiable insights."

    # Add context if available
    context_part = f"\nRelevant Context:\n{context}\n" if context else ""

    # Choose style dynamically
    if style == "formal":
        style_instruction = "Respond with structured, professional reasoning."
    elif style == "casual":
        style_instruction = "Explain in a friendly, simple way anyone can understand."
    else:
        style_instruction = "Respond clearly with a balance of depth and readability."

    # Final dynamic prompt
    prompt = f"""{base_instruction}
{style_instruction}
{context_part}
User Query: {user_input}
Answer:"""

    return prompt


def run():
    """
    Demo run for dynamic prompting.
    """
    print("⚡ Dynamic Prompting Module Running...")

    # Example query
    user_query = input("Enter your query: ")

    # For demo: mimic retrieved context
    fake_context = "Recent research suggests that misinformation often spreads faster than verified content."

    # Build a dynamic prompt
    prompt = build_dynamic_prompt(user_query, context=fake_context, style="balanced")

    # Call your LLM (through core/llm.py)
    response = llm.call_llm(prompt)

    print("\n=== Dynamic Prompt Response ===")
    print(response)