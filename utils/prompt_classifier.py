import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(project_root, 'prompt-injection', 'data', 'synthetic_healthcare_data.csv')
df = pd.read_csv(data_path)

def guardrailed_assistant(user_input):
    sensitive_keywords = [
        'name', 'diagnosis', 'medication', 'patient', 'ignore previous instructions',
        'export', 'show all', 'list all', 'record'
    ]
    if any(word in user_input.lower() for word in sensitive_keywords):
        return False, "Access denied: This request may expose sensitive patient information."
    return True, df['Name'].head(5).tolist()

def detect_iterative_attack(attempts):
    # If user makes 3+ blocked attempts in a row, flag as suspicious
    blocked_count = sum(1 for a in attempts if not a['success'])
    if blocked_count >= 3:
        return True
    return False

def classify_prompt(user_input):
    categories = {
        "TAP": ["ignore previous instructions", "disregard above", "pretend to be"],
        "PAIR": ["give me", "can you provide", "who are", "tell me about", "first detail"],
        "PAP": ["list all", "show all", "export", "record"],
    }
    user_input_lower = user_input.lower()
    for label, patterns in categories.items():
        for pattern in patterns:
            if pattern in user_input_lower:
                return label
    return "non-attack"

print("=== Gradient-Based Prompt Injection Attack Demo ===")
attempts = []
while True:
    prompt = input("Enter your prompt (or type 'exit' to stop): ")
    if prompt.lower() == 'exit':
        print("Demo ended.")
        break
    success, response = guardrailed_assistant(prompt)
    attempts.append({'prompt': prompt, 'success': success})
    if success:
        print("Bypassed guardrail! Assistant response:", response)
        break
    else:
        print("Blocked by guardrail:", response)
    if detect_iterative_attack(attempts):
        print("⚠️  Multiple blocked attempts detected! Possible iterative attack in progress.")