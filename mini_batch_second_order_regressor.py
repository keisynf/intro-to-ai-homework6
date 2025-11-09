# Mini batch second-order polynomial regressor
# By Keisy Núñez and Ezequiel Buck
# Following the examples in VehiclePrice_LinearRegression_GD3.py
# and Mnist_SoftmaxRegression_SGD3 seen in class

# Introduction to Machine Learning
# Vehicle Price dataset
# Linear regression solved through gradient descent in PyTorch
# Version 3: plot training and test MSE history
# By Juan Carlos Rojas
# Copyright 2025, Texas Tech University - Costa Rica

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use('TkAgg')  # To work on Linux


def collapse_small_categories(df, col, min_count=10, other_label="others"):
    counts = df[col].value_counts()
    rare = counts[counts < min_count].index
    df[col] = df[col].where(~df[col].isin(rare), other_label)
    return df


# Load and prepare the data (same preprocessing as the full-batch version)
df = pd.read_csv("vehicles_clean2.csv", header=0)
df = collapse_small_categories(df, "manufacturer", min_count=100)

# Store numerical column indices before one-hot encoding
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
if 'price' in numerical_cols:
    numerical_cols.remove('price')

# One-hot encode categorical columns
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

# Get indices of numerical columns in the final dataframe
numerical_indices = [i for i, col in enumerate(
    train_data.columns) if col in numerical_cols]

# Get sizes
ncoeffs = train_data.shape[1]
nsamples = train_data.shape[0]

# ============================================================
# Training constants (mini-batch)
# ============================================================
batch_size = 256      # chosen between 32 and 4096
learning_rate = 0.02
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
W = (torch.rand((ncoeffs, 1)) * 0.2 - 0.1).requires_grad_()
V = (torch.rand((len(numerical_indices), 1)) * 0.2 - 0.1).requires_grad_()
B = torch.zeros((1, 1), dtype=torch.float32)

# Ensure gradients are tracked
W.requires_grad_(True)
V.requires_grad_(True)
B.requires_grad_(True)

# Training loop (epochs + mini-batches)
train_cost_hist = []
test_cost_hist = []

start_time = time.perf_counter()
for epoch in range(1, n_epochs + 1):
    epoch_losses = []

    for X_batch, Y_batch in loader:
        # Forward (include second-order term)
        X_batch_squared = X_batch[:, numerical_indices] ** 2
        Y_pred_batch = X_batch @ W + X_batch_squared @ V + B
        mse_batch = torch.mean((Y_pred_batch - Y_batch) ** 2)

        # Backprop
        mse_batch.backward()

        # Gradient step
        with torch.no_grad():
            W -= learning_rate * W.grad
            V -= learning_rate * V.grad
            B -= learning_rate * B.grad

        # Zero grads
        W.grad.zero_()
        V.grad.zero_()
        B.grad.zero_()

        epoch_losses.append(mse_batch.item())

    # Record average training MSE for epoch
    avg_train_mse = float(np.mean(epoch_losses))
    train_cost_hist.append(avg_train_mse)

    # Evaluate on test set
    with torch.no_grad():
        X_test_squared = X_test[:, numerical_indices] ** 2
        Y_pred_test = X_test @ W + X_test_squared @ V + B
        mse_test = torch.mean((Y_pred_test - Y_test) ** 2).item()
        test_cost_hist.append(mse_test)

    if epoch % print_step == 0:
        print(
            f"Epoch {epoch:3d}: Train MSE: {avg_train_mse:.1f} Test MSE: {mse_test:.1f}")

end_time = time.perf_counter()
training_time = end_time - start_time
print(
    f"Total training time: {training_time:.3f} seconds ({training_time / n_epochs:.3f} s/epoch)")

# Stop gradient tracking on parameters
W.requires_grad_(False)
V.requires_grad_(False)
B.requires_grad_(False)

# Final RMSEs
train_rmse = (train_cost_hist[-1]) ** 0.5
test_rmse = (test_cost_hist[-1]) ** 0.5
print(f"Training RMSE: {train_rmse:.1f}")
print(f"Test RMSE: {test_rmse:.1f}")

# Plot MSE history across epochs (linear scale)
plt.plot(range(1, n_epochs + 1), train_cost_hist, "b", label="Train MSE")
plt.plot(range(1, n_epochs + 1), test_cost_hist, "r", label="Test MSE")
plt.xlabel("Epoch")
plt.ylabel("Cost (MSE)")
plt.title("Mini-batch training: Cost evolution")
plt.legend()
plt.show()
