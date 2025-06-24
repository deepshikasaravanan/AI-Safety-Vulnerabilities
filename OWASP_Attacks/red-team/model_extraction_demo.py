import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Train the victim model
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

class VictimNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        return self.fc(x)

victim_model = VictimNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(victim_model.parameters(), lr=0.01)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
for epoch in range(100):
    optimizer.zero_grad()
    outputs = victim_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# 2. Attacker queries the victim model (black-box access)
def query_victim(model, data):
    with torch.no_grad():
        logits = model(torch.tensor(data, dtype=torch.float32))
        preds = torch.argmax(logits, dim=1)
        return preds.numpy()

def query_victim_probs(model, data):
    with torch.no_grad():
        logits = model(torch.tensor(data, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1)
        return probs.numpy()

# Attacker uses test set as queries (could be random/public in real attack)
X_attack = X_test
y_attack = query_victim(victim_model, X_attack)  # Get victim's hard labels
y_attack_probs = query_victim_probs(victim_model, X_attack)  # Get victim's soft labels

# 3. Attacker trains their own (stolen) model on the queried data (hard labels)
class StolenNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        return self.fc(x)

stolen_model = StolenNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(stolen_model.parameters(), lr=0.01)
X_attack_tensor = torch.tensor(X_attack, dtype=torch.float32)
y_attack_tensor = torch.tensor(y_attack, dtype=torch.long)
for epoch in range(100):
    optimizer.zero_grad()
    outputs = stolen_model(X_attack_tensor)
    loss = criterion(outputs, y_attack_tensor)
    loss.backward()
    optimizer.step()

# 4. Enhanced attack: train with soft labels (probabilities)
stolen_soft_model = StolenNN()
soft_criterion = nn.KLDivLoss(reduction='batchmean')
soft_optimizer = optim.Adam(stolen_soft_model.parameters(), lr=0.01)
y_attack_probs_tensor = torch.tensor(y_attack_probs, dtype=torch.float32)
for epoch in range(100):
    soft_optimizer.zero_grad()
    outputs = stolen_soft_model(X_attack_tensor)
    log_probs = torch.log_softmax(outputs, dim=1)
    loss = soft_criterion(log_probs, y_attack_probs_tensor)
    loss.backward()
    soft_optimizer.step()

# 5. Evaluate all models
def evaluate(model, X, y):
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        preds = torch.argmax(logits, dim=1).numpy()
        return accuracy_score(y, preds)

print("Victim model accuracy on test set:", evaluate(victim_model, X_test, y_test))
print("Stolen model (hard labels) accuracy on test set:", evaluate(stolen_model, X_test, y_test))
print("Stolen model (soft labels) accuracy on test set:", evaluate(stolen_soft_model, X_test, y_test))

# 6. Detection: Watermarking
special_input = np.zeros((1, 64))  # Watermark trigger input
victim_output = query_victim(victim_model, special_input)
stolen_output = query_victim(stolen_model, special_input)
stolen_soft_output = query_victim(stolen_soft_model, special_input)
print("\nWatermarking Detection:")
print("Victim model output for special input:", victim_output)
print("Stolen model (hard labels) output for special input:", stolen_output)
print("Stolen model (soft labels) output for special input:", stolen_soft_output)
if (victim_output == stolen_output).all() or (victim_output == stolen_soft_output).all():
    print("Potential model theft detected: Stolen model mimics watermark response.")
else:
    print("No model theft detected based on watermark.")