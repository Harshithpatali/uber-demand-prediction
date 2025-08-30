# app.py - Streamlit app for surge multiplier prediction

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import datetime
import os

# Define the model class (same as in training)
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

# Check if required files exist
required_files = ['mean.npy', 'std.npy', 'feature_columns.pkl', 'surge_model.pth']
for file in required_files:
    if not os.path.exists(file):
        st.error(f"Error: {file} not found. Please run model.py to generate this file.")
        st.stop()

# Load saved items
mean = np.load('mean.npy')
std = np.load('std.npy')
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

input_size = len(feature_columns)
model = SurgePredictor(input_size)
model.load_state_dict(torch.load('surge_model.pth'))
model.eval()

# Unique values (from dataset)
cab_types = ['Uber', 'Lyft']
names = ['Black', 'Black SUV', 'Lux', 'Lux Black', 'Lux Black XL', 'Lyft', 'Lyft XL', 'Shared', 'Taxi', 'UberPool', 'UberX', 'UberXL', 'WAV']
locations = ['Back Bay', 'Beacon Hill', 'Boston University', 'Fenway', 'Financial District', 'Haymarket Square', 'North End', 'North Station', 'Northeastern University', 'South Station', 'Theatre District', 'West End']

# Streamlit UI
st.title('Uber/Lyft Surge Multiplier Prediction')

# Inputs
distance = st.number_input('Distance', min_value=0.0, value=1.24)
cab_type = st.selectbox('Cab Type', cab_types)
destination = st.selectbox('Destination', locations)
source = st.selectbox('Source', locations)
name = st.selectbox('Product Name', names)
temp = st.number_input('Temperature', value=40.81)
clouds = st.number_input('Clouds', min_value=0.0, max_value=1.0, value=0.89)
pressure = st.number_input('Pressure', value=1014.35)
rain = st.number_input('Rain', min_value=0.0, value=0.046227)
humidity = st.number_input('Humidity', min_value=0.0, max_value=1.0, value=0.93)
wind = st.number_input('Wind', min_value=0.0, value=1.36)

# Date and time input
date = st.date_input('Date')
time = st.time_input('Time')
time_stamp = datetime.datetime.combine(date, time)

# Create input DataFrame
input_data = {
    'distance': distance,
    'temp': temp,
    'clouds': clouds,
    'pressure': pressure,
    'rain': rain,
    'humidity': humidity,
    'wind': wind,
    'cab_type': cab_type,
    'destination': destination,
    'source': source,
    'name': name,
    'hour': time_stamp.hour,
    'day': time_stamp.day,
    'month': time_stamp.month,
    'weekday': time_stamp.weekday(),
    'is_weekend': 1 if time_stamp.weekday() >= 5 else 0
}
input_df = pd.DataFrame([input_data])

# One-hot encode with numeric dtype
input_df = pd.get_dummies(input_df, dtype=int)

# Align columns with training features
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# Scale
X_input = input_df.values.astype(float)
X_input = (X_input - mean) / std

# To tensor
X_tensor = torch.tensor(X_input, dtype=torch.float32)

# Predict
if st.button('Predict Surge Multiplier'):
    with torch.no_grad():
        prediction = model(X_tensor)
    st.write(f'Predicted Surge Multiplier: {prediction.item():.2f}')