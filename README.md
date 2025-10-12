# Hierarchical MPC for Hospital Microgrid Management

This project implements a hierarchical Model Predictive Control (MPC) system to optimize the energy management of a hospital microgrid. The controller's primary goal is to minimize daily operational costs while ensuring thermal comfort for occupants and robustly handling real-time power fluctuations.

The control architecture is split into two levels:
1.  **Level 1 (Supervisory MPC):** A long-horizon, slow-timescale economic planner that creates an optimal 24-hour energy schedule.
2.  **Level 2 (Real-Time MPC):** A short-horizon, fast-timescale controller that tracks the schedule from Level 1 and rejects unforeseen disturbances using a hybrid energy storage system.

---
## Key Features

* **Hierarchical Control Structure:** A two-level MPC for economic optimization and real-time disturbance rejection.
* **Integrated Electro-Thermal Model:** The controller co-optimizes HVAC usage for thermal comfort alongside electrical power dispatch, respecting temperature constraints.
* **Hybrid Energy Storage System (HESS):** Utilizes a high-energy Battery (BESS) for energy arbitrage and a high-power Supercapacitor (SC) to manage rapid power fluctuations.
* **ANN-Based Forecasting:** Employs a PyTorch-based Artificial Neural Network (ANN) to forecast outdoor temperature, solar PV generation, and uncontrollable electrical loads over a 24-hour horizon.
* **Detailed Cost Optimization:** The objective function minimizes a comprehensive cost model, including:
    * Time-of-Use (TOU) grid electricity prices.
    * A quadratic fuel cost model for the diesel generator (DG).
    * A linear cost model for battery degradation based on cycled energy.
* **Complete Simulation Environment:** The project includes scripts for generating training data, training the forecasting model, running the full 24-hour simulation, and visualizing the results.

---
## Project Structure

```
hospital_mpc/
├── config.py               # All system parameters and cost functions
├── forecasting.py          # Uses the trained ANN to provide forecasts
├── main_simulation.py      # Main script to run the simulation and plot results
├── models.py               # Defines the thermal, BESS, and SC models
├── mpc_controller.py       # Implements the Level 1 and Level 2 MPC logic
└── train_forecaster.py     # Generates synthetic data and trains the ANN model
```

---
## How to Run the Project

Follow these steps to set up and run the simulation.

### Step 1: Install Dependencies

First, ensure you have Python installed. Then, install the required libraries. It is recommended to use a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install numpy scipy matplotlib torch scikit-learn joblib
```

### Step 2: Train the Forecasting Model

The MPC relies on a pre-trained ANN model for its predictions. The training script generates synthetic data and saves the trained model and data scalers to the `hospital_mpc/` directory.

From your project's root directory, run the following command:

```bash
python -m hospital_mpc.train_forecaster
```

You should see output indicating the training progress, and upon completion, two files will be created: `hospital_mpc/forecast_model.pth` and `hospital_mpc/scalers.pkl`.

### Step 3: Run the Main Simulation

Once the forecasting model is trained, you can run the full 24-hour hierarchical MPC simulation.

```bash
python -m hospital_mpc.main_simulation
```

This command will:
1.  Load the trained ANN to get 24-hour forecasts.
2.  Run the Level 1 Supervisory MPC to generate the optimal economic plan.
3.  Execute the Level 2 Real-Time simulation loop, correcting for disturbances.
4.  Display plots visualizing the thermal performance, power dispatch, and HESS operation upon completion.