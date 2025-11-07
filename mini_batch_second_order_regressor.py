
import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def collapse_small_categories(df, col, min_count=10, other_label="others"):
    counts = df[col].value_counts()
    rare = counts[counts < min_count].index
    df[col] = df[col].where(~df[col].isin(rare), other_label)
    return df


# Load and prepare the data (same preprocessing as the full-batch version)
df = pd.read_csv("vehicles_clean2.csv", header=0)
df = collapse_small_categories(df, "manufacturer", min_count=100)

# Create squared columns for numerical columns
for col in df.select_dtypes(include=np.number).columns:
    df[col + "_squared"] = df[col] ** 2

df = pd.get_dummies(df, prefix_sep="_", drop_first=True, dtype=int)
labels = df["price"]
df = df.drop(columns="price")
train_data, test_data, train_labels, test_labels = sklearn.model_selection.train_test_split(
    df, labels, test_size=0.2, shuffle=True, random_state=2025)

# Standardize scale for all columns using train stats
train_means = train_data.mean()
train_stds = train_data.std()
train_data = (train_data - train_means) / train_stds
test_data = (test_data - train_means) / train_stds

# Get sizes
ncoeffs = train_data.shape[1]
nsamples = train_data.shape[0]

# ============================================================
# Training constants (mini-batch)
# ============================================================
batch_size = 256     # chosen between 32 and 4096
learning_rate = 0.01
n_epochs = 50
print_step = 1

# Convert data to PyTorch tensors
X = torch.tensor(train_data.values, dtype=torch.float32)
Y = torch.tensor(train_labels.values.reshape(-1, 1), dtype=torch.float32)

X_test = torch.tensor(test_data.values, dtype=torch.float32)
Y_test = torch.tensor(test_labels.values.reshape(-1, 1), dtype=torch.float32)

# Create DataLoader for mini-batches
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize weights and bias
torch.manual_seed(2025)
W = torch.rand((ncoeffs, 1), dtype=torch.float32) * 2 - 1
B = torch.zeros((1, 1), dtype=torch.float32)
W.requires_grad_(True)
B.requires_grad_(True)

# Training loop (epochs + mini-batches)
train_cost_hist = []
test_cost_hist = []

for epoch in range(1, n_epochs + 1):
    epoch_losses = []

    for X_batch, Y_batch in loader:
        # Forward
        Y_pred_batch = X_batch @ W + B
        mse_batch = torch.mean((Y_pred_batch - Y_batch) ** 2)

        # Backprop
        mse_batch.backward()

        # Gradient step
        with torch.no_grad():
            W -= learning_rate * W.grad
            B -= learning_rate * B.grad

        # Zero grads
        W.grad.zero_()
        B.grad.zero_()

        epoch_losses.append(mse_batch.item())

    # Record average training MSE for epoch
    avg_train_mse = float(np.mean(epoch_losses))
    train_cost_hist.append(avg_train_mse)

    # Evaluate on test set
    with torch.no_grad():
        Y_pred_test = X_test @ W + B
        mse_test = torch.mean((Y_pred_test - Y_test) ** 2).item()
        test_cost_hist.append(mse_test)

    if epoch % print_step == 0:
        print(
            f"Epoch {epoch:3d}: Train MSE: {avg_train_mse:.1f} Test MSE: {mse_test:.1f}")

# Stop gradient tracking on parameters
W.requires_grad_(False)
B.requires_grad_(False)

# Final RMSEs
train_rmse = (train_cost_hist[-1]) ** 0.5
test_rmse = (test_cost_hist[-1]) ** 0.5
print(f"Training RMSE: {train_rmse:.1f}")
print(f"Test RMSE: {test_rmse:.1f}")

# Plot MSE history across epochs
plt.plot(range(1, n_epochs + 1), train_cost_hist, "b", label="Train MSE")
plt.plot(range(1, n_epochs + 1), test_cost_hist, "r", label="Test MSE")
plt.xlabel("Epoch")
plt.ylabel("Cost (MSE)")
plt.title("Mini-batch training: Cost evolution")
plt.legend()
plt.show()
