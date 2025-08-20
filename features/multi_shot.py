# features/multi_shot.py
from core.llm import call_llm

def run():
    print("\n🚀 Running Multi-Shot Prompting...\n")

    # Multi-shot prompt with 3 examples
    prompt = """
You are an AI scam message classifier.
Classify the following message as either 'SCAM' or 'NOT SCAM'. 
Explain your reasoning briefly.

Examples:

1. Message: "Congratulations! You have won $1,000,000. Send your bank details to claim."
   Label: SCAM
   Reason: Asking for bank details with unrealistic reward.

2. Message: "Reminder: Your Amazon order #12345 will be delivered tomorrow."
   Label: NOT SCAM
   Reason: Normal order update, no sensitive info requested.

3. Message: "Urgent! Your account has been compromised. Verify immediately at http://fake-login.com"
   Label: SCAM
   Reason: Phishing attempt with fake login link.

---
Now classify this new message:

"Dear user, we noticed suspicious login attempts on your account. 
Please confirm your password here: http://security-check.com"
"""

    response = call_llm(prompt)
    print("🤖 LLM Response:\n")
    print(response)