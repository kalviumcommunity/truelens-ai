from groq import Groq
from utils.config import GROQ_API_KEY, GROQ_MODEL

def run():
    client = Groq(api_key=GROQ_API_KEY)

    prompt = (
        "Classify the following text as 'Authentic' or 'AI-generated' and explain why:\n\n"
        "This picturesque scene captures the ethereal charm of a sun-kissed Parisian morning."
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are an AI authenticity detection expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    print("\n=== Zero-Shot Classification Result ===")
    print(response.choices[0].message.content)  # <-- fixed here