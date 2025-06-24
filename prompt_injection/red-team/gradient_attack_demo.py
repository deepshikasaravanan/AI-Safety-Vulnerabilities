import pandas as pd
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = '/Users/Dell/Desktop/AI_Safety_Vulnerabilities_Demo/prompt_injection/data/synthetic_healthcare_data.csv'
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

def is_sensitive_prompt(user_input):
    sensitive_keywords = [
        "patient", "name", "full name", "list", "all", "every", "each",
        "diagnosis", "diagnoses", "condition", "disease", "illness",
        "medication", "medications", "drug", "prescription", "treatment",
        "export", "download", "dump", "print", "show", "display", "reveal",
        "record", "records", "history", "data", "details", "info", "information",
        "ignore previous instructions", "bypass", "override", "admin", "confidential",
        "private", "sensitive", "ssn", "social security", "address", "phone", "email"
    ]
    user_input_lower = user_input.lower()
    for keyword in sensitive_keywords:
        if keyword in user_input_lower:
            return True
    # Block common prompt injection tricks
    sensitive_patterns = [
        r"ignore.*instruction", r"disregard.*rule", r"forget.*previous", r"output.*all", r"give.*everything"
    ]
    for pattern in sensitive_patterns:
        if re.search(pattern, user_input_lower):
            return True
    return False

def secure_assistant(user_input):
    if is_sensitive_prompt(user_input):
        return "Sorry, I can't help with that."
    else:
        return "Your request does not match any sensitive pattern, but I still can't help with that."

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

# 1. Load and preprocess data
df = pd.read_csv('prompt_injection/data/synthetic_healthcare_data.csv')

# Assume 'Diagnosis' is the target, and all other columns are features
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Encode categorical features and target
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 2. Define and train a simple neural network
class EnhancedNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, num_classes)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

input_dim = X_train.shape[1]
num_classes = len(np.unique(y))
model = EnhancedNN(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
best_loss = float('inf')
patience = 10
counter = 0

for epoch in range(200):  # More epochs
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    # Early stopping (optional, if you have a validation set)
    if loss.item() < best_loss:
        best_loss = loss.item()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break

# 3. Evaluate on a test sample
model.eval()
idx = 0  # Pick the first test sample
x_orig = X_test_tensor[idx:idx+1]
y_true = y_test[idx]
with torch.no_grad():
    output = model(x_orig)
    pred = torch.argmax(output, dim=1).item()
print(f"Original prediction: {pred} (True label: {y_true})")

# 4. FGSM adversarial attack
epsilon = 0.1  # Attack strength
x_adv = x_orig.clone().detach().requires_grad_(True)
output = model(x_adv)
loss = criterion(output, torch.tensor([y_true]))
loss.backward()
# FGSM step
x_adv = x_adv + epsilon * x_adv.grad.sign()
with torch.no_grad():
    output_adv = model(x_adv)
    pred_adv = torch.argmax(output_adv, dim=1).item()
print(f"Adversarial prediction: {pred_adv} (True label: {y_true})")

if pred != pred_adv:
    print("Attack successful: Model prediction changed due to adversarial example!")
else:
    print("Attack failed: Model prediction did not change.")

with torch.no_grad():
    outputs = model(X_test_tensor)
    preds = torch.argmax(outputs, dim=1)
    accuracy = (preds == y_test_tensor).float().mean().item()
print(f"Test accuracy: {accuracy:.2f}")