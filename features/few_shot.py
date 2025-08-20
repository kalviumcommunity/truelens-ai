# features/few_shot.py

from groq import Groq

def run():
    client = Groq()

    # Few-shot prompt with multiple examples
    examples = """
    You are a scam detection assistant. Classify each message as 'Scam' or 'Safe'.

    Example 1:
    Message: "Congratulations! You won a free iPhone. Click here to claim."
    Label: Scam

    Example 2:
    Message: "Reminder: Your electricity bill of $120 is due tomorrow. Please pay to avoid late fees."
    Label: Safe

    Example 3:
    Message: "Dear customer, your account has been suspended. Verify your details at http://fakebank.com"
    Label: Scam

    Example 4:
    Message: "Your OTP for logging into NetBanking is 453219. Do not share this with anyone."
    Label: Safe
    """

    test_message = "Your credit card has been blocked. Click this link to verify your identity: http://fakeurl.com"

    prompt = f"""
    {examples}

    Now classify the following message:
    Message: "{test_message}"
    Label:
    """

    response = client.chat.completions.create(
        model="llama3-70b-8192",  # updated model
        messages=[{"role": "user", "content": prompt}],
    )

    print("\n=== Few-Shot Classification Result ===")
    print("Label:", response.choices[0].message.content.strip())