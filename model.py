# model.py - Feature engineering, encoding, scaling, and model training

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os

# Load the data
try:
    df = pd.read_csv(r'C:\Users\devar\Downloads\uber_demand_project\data\merged_rides_weather.csv', index_col='time_stamp', parse_dates=True)
    print("Data loaded successfully. Shape:", df.shape)
except Exception as e:
    print(f"Error loading CSV: {e}")
    raise

# 1. Feature Engineering
# Drop unnecessary columns
df = df.drop(['id', 'product_id'], axis=1)

# Check for missing values
print("Missing values before handling:\n", df.isna().sum())

# Handle missing values
numerical_cols = ['distance', 'temp', 'clouds', 'pressure', 'rain', 'humidity', 'wind', 'surge_multiplier']
categorical_cols = ['cab_type', 'destination', 'source', 'name']

# Impute numerical columns with median
for col in numerical_cols:
    if col in df.columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

# Drop rows with missing categorical values
df = df.dropna(subset=[col for col in categorical_cols if col in df.columns])

# Verify no remaining NaN values
print("Missing values after handling:\n", df.isna().sum())

# Extract time features from index
df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month
df['weekday'] = df.index.weekday
df['is_weekend'] = (df['weekday'] >= 5).astype(int)

# Ensure all numerical columns are numeric
for col in numerical_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().sum() > 0:
            print(f"Warning: Non-numeric values found in {col}. Filling NaNs with median.")
            df[col] = df[col].fillna(df[col].median())

# One-hot encode categorical columns with numeric dtype
df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], dtype=int)

# Verify all columns are numeric
print("Data types after encoding:\n", df.dtypes)
non_numeric_cols = df.dtypes[~df.dtypes.apply(lambda x: np.issubdtype(x, np.number))]
if not non_numeric_cols.empty:
    print("Warning: Non-numeric columns found after encoding:\n", non_numeric_cols)

# Separate target and features
target_col = 'surge_multiplier'
try:
    y = df[target_col].values
    X_df = df.drop([target_col], axis=1)
    X = X_df.values.astype(float)  # Ensure X is float
    print("Feature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)
except Exception as e:
    print(f"Error separating features and target: {e}")
    raise

# Verify X is numeric and handle NaNs
if not np.all(np.isfinite(X)):
    print("Warning: Non-finite values found in X. Replacing with 0.")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Scaling numerical features
mean = np.nanmean(X, axis=0)
std = np.nanstd(X, axis=0)
std = np.where(std == 0, 1e-8, std)  # Avoid division by zero
X = (X - mean) / std

# Final check for NaNs in scaled X
if np.any(np.isnan(X)):
    print("Warning: NaN values found in scaled features. Replacing with 0.")
    X = np.nan_to_num(X, nan=0.0)

# Save scaling parameters and feature columns
np.save('mean.npy', mean)
np.save('std.npy', std)
feature_columns = X_df.columns.tolist()
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

# Verify saved files
print("Saved files:")
for file in ['mean.npy', 'std.npy', 'feature_columns.pkl']:
    if os.path.exists(file):
        print(f"{file} saved successfully.")
    else:
        print(f"Error: {file} not found.")

# Convert to PyTorch tensors
try:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    print("Tensors created successfully. X_tensor shape:", X_tensor.shape, "y_tensor shape:", y_tensor.shape)
except Exception as e:
    print(f"Error creating tensors: {e}")
    raise

# Create DataLoader
try:
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print("DataLoader created successfully.")
except Exception as e:
    print(f"Error creating DataLoader: {e}")
    raise

# Define the model
class SurgePredictor(nn.Module):
    def __init__(self, input_size):
        super(SurgePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

try:
    input_size = X.shape[1]
    model = SurgePredictor(input_size)
    print("Model initialized with input size:", input_size)
except Exception as e:
    print(f"Error initializing model: {e}")
    raise

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
try:
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
except Exception as e:
    print(f"Error during training: {e}")
    raise

# Save the model
try:
    torch.save(model.state_dict(), 'surge_model.pth')
    print("Model saved as surge_model.pth")
    if os.path.exists('surge_model.pth'):
        print("surge_model.pth saved successfully.")
    else:
        print("Error: surge_model.pth not found.")
except Exception as e:
    print(f"Error saving model: {e}")
    raise