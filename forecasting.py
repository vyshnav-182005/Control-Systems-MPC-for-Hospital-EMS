# hospital_mpc/forecasting.py

import torch
import torch.nn as nn
import numpy as np
import joblib
from . import config

# --- 1. Define the same ANN Architecture used for training ---
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

# --- 2. Load the trained model and scalers ---
try:
    model = ForecastANN()
    model.load_state_dict(torch.load('hospital_mpc/forecast_model.pth'))
    model.eval()  # Set the model to evaluation mode
    scaler_X, scaler_y = joblib.load('hospital_mpc/scalers.pkl')
    print("🤖 PyTorch ANN forecasting model loaded successfully.")
except FileNotFoundError:
    print("⚠️ Model files not found. Please run 'train_forecaster.py' first.")
    model = None
    scaler_X, scaler_y = None, None

def get_ann_forecasts():
    """
    Generates 24-hour forecasts using the pre-trained ANN model.
    """
    if model is None:
        raise RuntimeError("Forecasting model is not available. Run training first.")

    # Prepare the input data for the next 24 hours
    steps = config.SUPERVISORY_STEPS
    minutes = np.arange(steps) * config.SUPERVISORY_TIMESTEP_MIN
    hours = (minutes / 60.0).reshape(-1, 1)

    # Scale the input and convert to tensor
    hours_scaled = scaler_X.transform(hours)
    hours_tensor = torch.FloatTensor(hours_scaled)

    # Get predictions from the model
    with torch.no_grad(): # No need to track gradients for inference
        predictions_scaled = model(hours_tensor)

    # Inverse transform the predictions to get real values
    predictions = scaler_y.inverse_transform(predictions_scaled.numpy())

    # The model predicts the 3 dynamic variables
    t_out_forecast = predictions[:, 0]
    pv_forecast = np.maximum(0, predictions[:, 1]) # Ensure PV is not negative
    uncontrollable_load_forecast = predictions[:, 2]

    # Internal gain is still treated as a constant forecast
    internal_gain_kw = np.full(steps, 15.0)

    return {
        't_out_c': t_out_forecast,
        'pv_kw': pv_forecast,
        'uncontrollable_load_kw': uncontrollable_load_forecast,
        'internal_gain_kw': internal_gain_kw
    }

def get_mathematical_forecasts():
    """This function is no longer the primary source for forecasts but is kept
       for reference or fallback if needed."""
    steps = config.SUPERVISORY_STEPS
    minutes = np.arange(steps) * config.SUPERVISORY_TIMESTEP_MIN
    hours = minutes / 60.0
    t_out_forecast = 15 + 10 * np.sin((hours - 8) * np.pi / 12)
    pv_forecast = np.maximum(0, 75 * np.sin((hours - 6) * np.pi / 12) * (1 - (hours - 13)**2 / 100))
    base_load = 50
    peak_load = 100
    uncontrollable_load_forecast = base_load + (peak_load - base_load) * (
        0.5 * (1 + np.sin((hours - 9) * np.pi / 12))
    )
    internal_gain_kw = np.full(steps, 15.0)
    return {
        't_out_c': t_out_forecast,
        'pv_kw': pv_forecast,
        'uncontrollable_load_kw': uncontrollable_load_forecast,
        'internal_gain_kw': internal_gain_kw
    }

def get_real_time_measurements(forecasts, supervisory_step, realtime_step):
    """
    Simulates real-time measurements by adding noise to forecasts.
    Interpolates 15-min forecasts to 1-min resolution.
    """
    interp_factor = realtime_step / config.REALTIME_STEPS_PER_SUPERVISORY_STEP
    
    def interpolate(data, step):
        if step + 1 < len(data):
            return data[step] * (1 - interp_factor) + data[step+1] * interp_factor
        return data[step]

    pv_forecast_interp = interpolate(forecasts['pv_kw'], supervisory_step)
    load_forecast_interp = interpolate(forecasts['uncontrollable_load_kw'], supervisory_step)
    
    # Add random noise to simulate forecast error
    pv_real = pv_forecast_interp * (1 + (np.random.rand() - 0.5) * 0.2)
    load_real = load_forecast_interp * (1 + (np.random.rand() - 0.5) * 0.1)
    
    # Add high-frequency noise for supercapacitor to handle
    load_real += 15 * np.sin(realtime_step * np.pi) # Simulates MRI-like spikes
    
    return {'pv_kw': pv_real, 'uncontrollable_load_kw': load_real}