import pandas as pd
import os

# Load data
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(project_root, 'prompt_injection', 'data', 'synthetic_healthcare_data.csv')
df = pd.read_csv(data_path)

def model_inversion_assistant(user_input):
    user_input_lower = user_input.lower()
    # Enhanced inversion queries
    if "most common name" in user_input_lower:
        return df['Name'].value_counts().idxmax()
    if "most common diagnosis" in user_input_lower:
        return df['Diagnosis'].value_counts().idxmax()
    if "first letter of the most common name" in user_input_lower:
        return df['Name'].value_counts().idxmax()[0]
    if "length of the most common name" in user_input_lower:
        return len(df['Name'].value_counts().idxmax())
    if "names starting with" in user_input_lower:
        letter = user_input_lower.split("names starting with")[-1].strip().upper()[0]
        return df[df['Name'].str.startswith(letter)]['Name'].unique().tolist()
    if "diagnosis for" in user_input_lower:
        # e.g., "diagnosis for John Smith"
        name = user_input_lower.split("diagnosis for")[-1].strip().title()
        diagnoses = df[df['Name'] == name]['Diagnosis'].unique().tolist()
        return diagnoses if diagnoses else "No record found."
    if "what names appear" in user_input_lower or "list unique names" in user_input_lower:
        return df['Name'].unique().tolist()
    if "correlation between name and diagnosis" in user_input_lower:
        # Show top name-diagnosis pairs
        return df.groupby(['Name', 'Diagnosis']).size().sort_values(ascending=False).head(3)
    # Normal guardrail logic
    sensitive_keywords = [
        'all patient names', 'all diagnoses', 'export', 'ignore previous instructions'
    ]
    if any(word in user_input_lower for word in sensitive_keywords):
        return "Access denied: This request may expose sensitive patient information."
    return "Sorry, I can't help with that."

def detect_inversion_attack(user_input, response):
    # Flag if the response reconstructs sensitive data via statistics or unique values
    suspicious = False
    if isinstance(response, list) and len(response) > 0:
        suspicious = True
    if isinstance(response, str) and response in df['Name'].tolist():
        suspicious = True
    if isinstance(response, pd.Series) and not response.empty:
        suspicious = True
    if suspicious:
        print("⚠️  Possible model inversion attack detected! Review this prompt/response pair.")




attempt_log = []
while True:
    prompt = input("Enter your prompt: ")
    if prompt.lower() == 'exit':
        print("Demo ended.")
        break
    response = model_inversion_assistant(prompt)
    print("Assistant response:", response)
    detect_inversion_attack(prompt, response)
    attempt_log.append({'prompt': prompt, 'response': str(response)})

# Print summary log
print("\n=== Attack Attempt Log ===")
for entry in attempt_log:
    print(f"Prompt: {entry['prompt']} --> Response: {entry['response']}")