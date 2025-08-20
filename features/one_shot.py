from groq import Groq
from utils.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def run():
    # Define the example and the test message
    example_message = "Congratulations! You've won a free iPhone, click here to claim."
    example_label = "Scam"

    test_message = "Your OTP for login is 492381."

    prompt = f"""
You are a cybersecurity assistant. Classify messages as "Scam" or "Safe".

Example:
Message: "{example_message}"
Label: {example_label}

Now classify:
Message: "{test_message}"
Label:
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print("\n=== One-Shot Classification Result ===")
    print(response.choices[0].message.content.strip())