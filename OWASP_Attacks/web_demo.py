import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.title("PGD (Projected Gradient Descent) Adversarial Attack Demo")

# Load and preprocess data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, 'prompt_injection', 'data', 'synthetic_healthcare_data.csv')
df = pd.read_csv(data_path)

X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Encode categorical features and target
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define and train model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_dim = X_train.shape[1]
num_classes = len(np.unique(y))
model = SimpleNN(input_dim, num_classes)
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

# PGD attack function
def pgd_attack(model, x, y, epsilon=0.1, alpha=0.01, num_iter=40):
    x_adv = x.clone().detach()
    x_adv.requires_grad = True
    for _ in range(num_iter):
        output = model(x_adv)
        loss = criterion(output, torch.tensor([y]))
        loss.backward()
        # Take a small step in the direction of the gradient
        x_adv = x_adv + alpha * x_adv.grad.sign()
        # Project back into the epsilon-ball of the original input
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, -3, 3)  # adjust clamp as needed for your data
        x_adv = x_adv.detach()
        x_adv.requires_grad = True
    return x_adv

# Add this FGSM attack function
def fgsm_attack(model, x, y, epsilon=0.1):
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = criterion(output, torch.tensor([y]))
    loss.backward()
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv, -3, 3)  # adjust clamp as needed for your data
    return x_adv

# Streamlit UI for PGD attack
idx = st.number_input("Select test sample index", min_value=0, max_value=len(X_test_tensor)-1, value=0, step=1)
epsilon = st.slider("Epsilon (attack strength)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
alpha = st.slider("Alpha (step size)", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
num_iter = st.slider("Number of PGD steps", min_value=1, max_value=100, value=40, step=1)

if st.button("Run PGD & FGSM Attack"):
    x_orig = X_test_tensor[idx:idx+1]
    y_true = y_test[idx]
    with torch.no_grad():
        output = model(x_orig)
        orig_pred = torch.argmax(output, dim=1).item()

    # FGSM attack
    x_fgsm = fgsm_attack(model, x_orig, y_true, epsilon=epsilon)
    with torch.no_grad():
        output_fgsm = model(x_fgsm)
        fgsm_pred = torch.argmax(output_fgsm, dim=1).item()

    # PGD attack
    x_pgd = pgd_attack(model, x_orig, y_true, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
    with torch.no_grad():
        output_pgd = model(x_pgd)
        pgd_pred = torch.argmax(output_pgd, dim=1).item()

    st.write(f"**Original prediction:** {orig_pred} (True label: {y_true})")
    st.write(f"**FGSM prediction:** {fgsm_pred}")
    st.write(f"**PGD prediction:** {pgd_pred}")

    if orig_pred != fgsm_pred:
        st.success("FGSM attack successful: Model prediction changed!")
    else:
        st.warning("FGSM attack failed: Model prediction did not change.")

    if orig_pred != pgd_pred:
        st.success("PGD attack successful: Model prediction changed!")
    else:
        st.warning("PGD attack failed: Model prediction did not change.")

    # Visualization
    orig_np = x_orig.detach().cpu().numpy().flatten()
    fgsm_np = x_fgsm.detach().cpu().numpy().flatten()
    pgd_np = x_pgd.detach().cpu().numpy().flatten()
    fgsm_perturb = fgsm_np - orig_np
    pgd_perturb = pgd_np - orig_np

    st.subheader("Adversarial Perturbation Visualization")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(orig_np, label="Original", marker='o')
    ax.plot(fgsm_np, label="FGSM", marker='x')
    ax.plot(pgd_np, label="PGD", marker='^')
    ax.plot(fgsm_perturb, label="FGSM Perturbation", linestyle='dashed')
    ax.plot(pgd_perturb, label="PGD Perturbation", linestyle='dashed')
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)