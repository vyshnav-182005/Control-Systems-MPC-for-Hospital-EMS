# hospital_mpc/config.py

import numpy as np

# --- Simulation Parameters ---
SIMULATION_HOURS = 24
SUPERVISORY_TIMESTEP_MIN = 15  # Level 1 MPC runs every 15 minutes
REALTIME_TIMESTEP_MIN = 1      # Level 2 MPC runs every 1 minute

SUPERVISORY_STEPS = SIMULATION_HOURS * (60 // SUPERVISORY_TIMESTEP_MIN)
REALTIME_STEPS_PER_SUPERVISORY_STEP = SUPERVISORY_TIMESTEP_MIN // REALTIME_TIMESTEP_MIN
TOTAL_REALTIME_STEPS = SIMULATION_HOURS * 60

# --- Thermal Model Parameters (RC Model) ---
C_MASS_KWH_PER_C = 15.0
R_WALL_C_PER_KW = 2.0
T_ZONE_MIN_C = 22.0
T_ZONE_MAX_C = 25.0
HVAC_COP = 3.0

# --- Electrical Asset Parameters ---
# BESS Parameters (High Energy)
BESS_CAPACITY_KWH = 200.0
BESS_MAX_POWER_KW = 50.0
BESS_INITIAL_SOC = 0.5
BESS_EFFICIENCY = 0.95

# Supercapacitor Parameters (High Power)
SC_CAPACITY_KWH = 5.0
SC_MAX_POWER_KW = 100.0
SC_INITIAL_SOC = 0.5
SC_EFFICIENCY = 0.98

# DG Parameters
DG_MAX_POWER_KW = 100.0

# --- Cost Parameters ---
# Time-of-Use (ToU) grid pricing ($/kWh) for 15-minute intervals
GRID_PRICE_TOU_HOURLY = np.array([
    0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.15, 0.15, 0.20, 0.20, 0.20, 0.25,
    0.25, 0.25, 0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10
])
GRID_PRICE_TOU = np.repeat(GRID_PRICE_TOU_HOURLY, 4) # Repeat for 15-min intervals

# DG quadratic fuel cost: C(P) = a*P^2 + b*P + c ($/hr)
DG_COST_A = 0.001
DG_COST_B = 0.02
DG_COST_C = 0.5

# Battery degradation cost ($/kWh cycled)
BATT_DEG_COST_PER_KWH = 0.05

# Penalty for violating thermal comfort bands ($ per °C)
THERMAL_PENALTY_WEIGHT = 1000.0
