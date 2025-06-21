import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 1. Load and split data
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 2. Define and train a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        return self.fc(x)

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# 3. Get model confidence scores
def get_confidence(model, data):
    with torch.no_grad():
        logits = model(torch.tensor(data, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1)
        max_conf, _ = torch.max(probs, dim=1)
        return max_conf.numpy()

train_conf = get_confidence(model, X_train)
test_conf = get_confidence(model, X_test)

print("Train confidence mean:", train_conf.mean())
print("Test confidence mean:", test_conf.mean())

# 4. Membership Inference Attack using a simple threshold
threshold = (train_conf.mean() + test_conf.mean()) / 2
train_pred = (train_conf > threshold).astype(int)
test_pred = (test_conf > threshold).astype(int)
train_true = np.ones_like(train_pred)
test_true = np.zeros_like(test_pred)
accuracy = np.mean(np.concatenate([train_pred, test_pred]) == np.concatenate([train_true, test_true]))
print(f"Simple threshold attack accuracy: {accuracy:.2f}")

# 5. Enhanced attack: train a classifier on confidence scores
X_attack = np.concatenate([train_conf.reshape(-1,1), test_conf.reshape(-1,1)])
y_attack = np.concatenate([np.ones_like(train_conf), np.zeros_like(test_conf)])
attack_model = RandomForestClassifier()
attack_model.fit(X_attack, y_attack)
print("Attack model accuracy:", attack_model.score(X_attack, y_attack))

def user_membership_inference(model, attack_model, threshold):
    print("\nEnter 64 pixel values (0-16) separated by spaces (for a digit image):")
    user_input = input()
    try:
        user_data = np.array([float(x) for x in user_input.strip().split()])
        if user_data.shape[0] != 64:
            print("Error: Please enter exactly 64 values.")
            return
        user_tensor = torch.tensor(user_data.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            logits = model(user_tensor)
            probs = torch.softmax(logits, dim=1)
            max_conf, _ = torch.max(probs, dim=1)
            confidence = max_conf.item()
        print(f"Model confidence: {confidence:.4f}")

        # Simple threshold attack
        is_member_simple = int(confidence > threshold)
        print(f"Simple threshold attack: {'Member' if is_member_simple else 'Non-member'}")

        # Enhanced attack
        is_member_enhanced = attack_model.predict([[confidence]])[0]
        print(f"Attack model prediction: {'Member' if is_member_enhanced else 'Non-member'}")
    except Exception as e:
        print("Invalid input:", e)

if __name__ == "__main__":
    user_membership_inference(model, attack_model, threshold)