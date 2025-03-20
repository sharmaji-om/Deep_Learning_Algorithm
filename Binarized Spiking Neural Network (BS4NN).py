import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import spikegen
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# Set the number of K folds
K_FOLDS = 2

# Read the dataset
df = pd.read_csv(r"E:\part-00001_preprocessed_dataset.csv")
df = df.sample(frac=0.2, random_state=42)
df.rename(columns={df.columns[-1]: 'label'}, inplace=True)

X = df.drop(columns=['label']).values
y = df['label'].values

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)

# Define Spiking Neural Network (BS4NN) with Binarized Weights
class BS4NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=0.9)
    
    def forward(self, x):
        spk_rec = []
        for step in range(10):  # Simulate over time
            cur = self.fc1(x)
            spk, _ = self.lif1(cur)
            cur = self.fc2(spk)
            spk, _ = self.lif2(cur)
            spk_rec.append(spk)
        return torch.stack(spk_rec).mean(0)  # Average over time

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
    y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    # Initialize Model
    model = BS4NN(input_size=X.shape[1], hidden_size=64, output_size=len(np.unique(y)))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Train Model
    model.train()
    for epoch in range(10):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
    
    # Evaluate Model
    model.eval()
    y_preds, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            y_pred = torch.argmax(output, dim=1)
            y_preds.extend(y_pred.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())
    
    acc = accuracy_score(y_true, y_preds)
    cm = confusion_matrix(y_true, y_preds)
    results.append({'Fold': fold + 1, 'Accuracy': acc})
    
    # Save confusion matrix
    os.makedirs("confusion_matrices", exist_ok=True)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'BS4NN - Fold {fold+1} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrices/BS4NN_fold_{fold+1}.png')
    plt.close()

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("bs4nn_metrics.csv", index=False)
print("BS4NN Classification Metrics:")
print(results_df)
