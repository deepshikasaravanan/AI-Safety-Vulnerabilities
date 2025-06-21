import pandas as pd
import os

# Load data
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
    blocked_count = sum(1 for a in attempts if not a['success'])
    return blocked_count >= 3

print("=== Interactive Gradient-Based Prompt Injection Attack Demo ===")
print("Try to bypass the guardrail by refining your prompt. Type 'exit' to stop.\n")

attempt = 1
attempts = []
while True:
    prompt = input(f"Attempt {attempt} - Enter your prompt: ")
    if prompt.lower() == 'exit':
        print("Demo ended.")
        break
    success, response = guardrailed_assistant(prompt)
    attempts.append({'attempt': attempt, 'prompt': prompt, 'success': success, 'response': response})
    if success:
        print("Bypassed guardrail! Assistant response:", response)
        break
    else:
        print("Blocked by guardrail:", response)
    if detect_iterative_attack(attempts):
        print("⚠️  Multiple blocked attempts detected! Possible iterative attack in progress.")
    attempt += 1

# Print a summary log at the end
print("\n=== Attack Attempt Log ===")
for entry in attempts:
    status = "Bypassed" if entry['success'] else "Blocked"
    print(f"Attempt {entry['attempt']}: {entry['prompt']} --> {status}")