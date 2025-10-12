# hospital_mpc/train_forecaster.py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

from . import config

# --- 1. Define the ANN Architecture ---
# This class needs to be accessible by both the training script and the forecasting script.
class ForecastANN(nn.Module):
    def __init__(self):
        super(ForecastANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 128),  # Input: hour of the day
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)   # Output: t_out, pv_kw, uncontrollable_load
        )

    def forward(self, x):
        return self.network(x)

# --- 2. Generate Synthetic Training Data ---
def generate_training_data(days=100):
    """Generates a larger dataset using the original mathematical model for training."""
    total_steps = config.SUPERVISORY_STEPS * days
    minutes = np.arange(total_steps) * config.SUPERVISORY_TIMESTEP_MIN
    hours = (minutes / 60.0) % 24  # Ensure hours loop every 24h

    # Outdoor Temperature (°C)
    t_out = 15 + 10 * np.sin((hours - 8) * np.pi / 12) + np.random.randn(total_steps) * 0.5
    
    # PV Generation (kW)
    pv = np.maximum(0, 75 * np.sin((hours - 6) * np.pi / 12) * (1 - (hours - 13)**2 / 100))
    pv += np.maximum(0, np.random.randn(total_steps) * 3) # Add noise only when > 0
    
    # Uncontrollable Loads (kW)
    base_load = 50
    peak_load = 100
    load = base_load + (peak_load - base_load) * (0.5 * (1 + np.sin((hours - 9) * np.pi / 12)))
    load += np.random.randn(total_steps) * 2.5

    # Combine into features (X) and targets (y)
    X = hours.reshape(-1, 1)
    y = np.vstack([t_out, pv, load]).T
    
    return X, y

# --- 3. Training Script ---
def train_model():
    """Trains the ANN and saves the model and scalers."""
    print("🧠 Starting ANN training...")
    
    # Generate data
    X_train, y_train = generate_training_data()
    
    # Scale data for better training performance
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_train_scaled)
    y_tensor = torch.FloatTensor(y_train_scaled)
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Initialize model, loss, and optimizer
    model = ForecastANN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 200
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

    # Save the trained model and scalers
    torch.save(model.state_dict(), 'hospital_mpc/forecast_model.pth')
    joblib.dump((scaler_X, scaler_y), 'hospital_mpc/scalers.pkl')
    
    print("✅ Training complete. Model and scalers saved.")

if __name__ == '__main__':
    train_model()