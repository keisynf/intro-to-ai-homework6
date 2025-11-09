# Mini batch second-order polynomial regressor
# By Keisy Núñez and Ezequiel Buck
# Following the examples in VehiclePrice_LinearRegression_GD3.py
# and Mnist_LassoRegression1 seen in class

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

matplotlib.use('TkAgg')  # To work on Linux


def collapse_small_categories(df, col, min_count=10, other_label="others"):
    counts = df[col].value_counts()
    rare = counts[counts < min_count].index
    df[col] = df[col].where(~df[col].isin(rare), other_label)
    return df


# Load and prepare the data
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

# Standardize scale for all columns
train_means = train_data.mean()
train_stds = train_data.std()
train_data = (train_data - train_means) / train_stds
test_data = (test_data - train_means) / train_stds

# Get indices of numerical columns in the final dataframe
numerical_indices = [i for i, col in enumerate(
    train_data.columns) if col in numerical_cols]

# Get some lengths
ncoeffs = train_data.shape[1]
nsamples = train_data.shape[0]

# ============================================================
# Training constants
# ============================================================
learning_rate = 0.02
n_iterations = 10000
lambda_reg = 0.001  # L2 regularization strength

# ============================================================
# Convert data to PyTorch tensors
# ============================================================
X = torch.tensor(train_data.values, dtype=torch.float32)
Y = torch.tensor(train_labels.values.reshape(-1, 1), dtype=torch.float32)

X_test = torch.tensor(test_data.values, dtype=torch.float32)
Y_test = torch.tensor(test_labels.values.reshape(-1, 1), dtype=torch.float32)

# ============================================================
# Create and initialize weights and bias
# ============================================================

W = (torch.rand((ncoeffs, 1)) * 0.2 - 0.1).requires_grad_()
V = (torch.rand((len(numerical_indices), 1)) * 0.2 - 0.1).requires_grad_()

# Create bias
B = torch.zeros((1, 1), dtype=torch.float32)

# Start tracking gradients on W
W.requires_grad_(True)
V.requires_grad_(True)
B.requires_grad_(True)

# ============================================================
# Training loop
# ============================================================

# History lists
train_cost_hist = []
test_cost_hist = []
eval_step = 100     # evaluate and record MSE every eval_step iterations

# Start timing
start_time = time.time()

for iteration in range(n_iterations):

    # Forward pass: predictions (include second-order term)
    X_squared = X[:, numerical_indices] ** 2
    Y_pred = X @ W + X_squared @ V + B

    # Mean squared error
    mse = torch.mean((Y_pred - Y) ** 2)

    # L2 regularization on W and V
    l2_reg = lambda_reg * (torch.sum(W ** 2) + torch.sum(V ** 2))

    # Total cost
    total_cost = mse + l2_reg

    # Compute gradients of total cost with respect to W, V & B
    total_cost.backward()

    # Gradient descent step
    with torch.no_grad():
        W -= learning_rate * W.grad
        V -= learning_rate * V.grad
        B -= learning_rate * B.grad

    # Zero gradients before next step
    W.grad.zero_()
    V.grad.zero_()
    B.grad.zero_()

    # Evaluate and record cost every eval_step iterations
    if iteration % eval_step == 0:
        # Training cost (MSE + L2)
        mse_train = mse.item()
        l2_cost = l2_reg.item()
        total_train_cost = mse_train + l2_cost
        train_cost_hist.append(total_train_cost)

        # Test cost (MSE + L2)
        with torch.no_grad():
            X_test_squared = X_test[:, numerical_indices] ** 2
            Y_pred_test = X_test @ W + X_test_squared @ V + B
            mse_test = torch.mean((Y_pred_test - Y_test) ** 2).item()
            total_test_cost = mse_test + l2_cost
            test_cost_hist.append(total_test_cost)

        print(f"Iteration {iteration:4d}: Train Cost: {total_train_cost:.1f} (MSE: {mse_train:.1f}, L2: {l2_cost:.3f}) Test Cost: {total_test_cost:.1f}")

# Stop timing
end_time = time.time()
training_time = end_time - start_time

# Stop tracking gradients on W & B
W.requires_grad_(False)
V.requires_grad_(False)
B.requires_grad_(False)

# Print the final MSEs and training time
print(f"Training MSE: {train_cost_hist[-1]:.1f}")
print(f"Test MSE: {test_cost_hist[-1]:.1f}")
print(f"Training RMSE: {train_cost_hist[-1]**0.5:.1f}")
print(f"Test RMSE: {test_cost_hist[-1]**0.5:.1f}")
print(f"Training time: {training_time:.2f} seconds")

# Compute standard deviation of model coefficients
with torch.no_grad():
    W_vals = W.flatten()
    V_vals = V.flatten()
    W_std = torch.std(W_vals).item()
    V_std = torch.std(V_vals).item()
    combined_std = torch.std(torch.cat([W_vals, V_vals])).item()

print(f"Std dev of W coefficients: {W_std:.6f}")
print(f"Std dev of V coefficients: {V_std:.6f}")
print(f"Std dev of all coefficients (W+V): {combined_std:.6f}")

# Plot MSE history
iterations_hist = [i for i in range(0, n_iterations, eval_step)]
plt.plot(iterations_hist, train_cost_hist, "b", label="Train MSE")
plt.plot(iterations_hist, test_cost_hist, "r", label="Test MSE")
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.title("Cost evolution")
plt.legend()
plt.show()
