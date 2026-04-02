# hospital_mpc/main_simulation_ss.py

import matplotlib.pyplot as plt
import numpy as np
import time

# These imports assume your project structure is correct
from . import config
from .forecasting import get_ann_forecasts
from .models import ThermalZoneModel, BESSModel, SCModel
from .fuzzypid_controller import FuzzySupervisory, PIDRealTime

import matplotlib.pyplot as plt

# ================= GLOBAL PLOT SETTINGS (IMPORTANT) =================
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 17,          # base text
    "axes.labelsize": 17,     # X/Y labels
    "axes.titlesize": 17,     # titles
    "legend.fontsize": 17,    # legend
    "xtick.labelsize": 17,    # tick labels
    "ytick.labelsize": 17,
    "figure.dpi": 140      # clarity when zoomed
})

plt.show()


def run_simulation_ss():
    """Main function to run the hierarchical electro-thermal MPC simulation."""
    start_time = time.time()
    
    # 1. Get 24-hour Forecasts using the ANN
    try:
        forecasts = get_ann_forecasts()
    except (RuntimeError, FileNotFoundError) as e:
        print(f"Error getting forecasts: {e}")
        print("Please ensure the forecasting model is trained by running train_forecaster.py")
        return

    # 2. Run Level 1 Supervisory MPC to get the optimal 24-hour plan
    sup_mpc = FuzzySupervisory(forecasts)
    optimal_plan_u = sup_mpc.create_optimal_plan()
    
    if optimal_plan_u is None:
        print("Halting simulation due to optimization failure.")
        return

    p_grid_plan, p_dg_plan, p_bess_plan, q_hvac_plan = optimal_plan_u.T
    p_hvac_plan = q_hvac_plan / config.HVAC_COP
    
    # 3. Initialize Level 2 MPC and system models
    rt_mpc = PIDRealTime()
    thermal_zone = ThermalZoneModel(initial_temp_c=23.5)
    bess = BESSModel()
    sc = SCModel()

    # History dictionary with ALL keys for plotting
    history = {
        't_in_c': [], 'bess_soc': [], 'sc_soc': [], 'p_grid_kw': [], 
        'p_dg_kw': [], 'p_bess_kw': [], 'p_sc_kw': [], 'p_pv_kw': [], 
        'p_load_kw': [], 'cost': [],
        # Keys for advanced plots
        'power_error_kw': [],
        'p_bess_correction': [],
        'p_sc_correction': [],
        'p_bess_plan_interp': []
    }

    print("\n⚡️ Starting Level 2 real-time simulation loop (1-min steps)...")
    cumulative_cost = 0
    forecast_minutes = np.arange(config.SUPERVISORY_STEPS) * config.SUPERVISORY_TIMESTEP_MIN
    
    # --- Main Simulation Loop ---
    for k_sup in range(config.SUPERVISORY_STEPS):
        for k_rt in range(config.REALTIME_STEPS_PER_SUPERVISORY_STEP):
            k_total = k_sup * config.REALTIME_STEPS_PER_SUPERVISORY_STEP + k_rt
            
            # Simulate real measurements with noise
            pv_real = forecasts['pv_kw'][k_sup] * (1 + (np.random.rand() - 0.5) * 0.2)
            load_real = forecasts['uncontrollable_load_kw'][k_sup] * (1 + (np.random.rand() - 0.5) * 0.1)
            load_real += 15 * np.sin(k_total * np.pi * 0.2) # High freq noise
            
            # Calculate power error between plan and reality
            planned_net_load = forecasts['uncontrollable_load_kw'][k_sup] + p_hvac_plan[k_sup] - forecasts['pv_kw'][k_sup]
            real_net_load = load_real + p_hvac_plan[k_sup] - pv_real
            power_error_kw = real_net_load - planned_net_load
            
            # Get real-time corrections from Level 2 MPC
            p_bess_correction, p_sc_correction, p_grid_correction = rt_mpc.dispatch(power_error_kw)
            
            # Determine actual power flows
            p_bess_actual = p_bess_plan[k_sup] + p_bess_correction
            p_sc_actual = p_sc_correction
            p_dg_actual = p_dg_plan[k_sup]
            p_grid_actual = p_grid_plan[k_sup] + p_grid_correction
            
            # Update system models with actual power flows
            t_out_interp = np.interp(k_total, forecast_minutes, forecasts['t_out_c'])
            internal_gain_interp = np.interp(k_total, forecast_minutes, forecasts['internal_gain_kw'])
            t_in = thermal_zone.step(t_out_interp, internal_gain_interp, q_hvac_plan[k_sup])
            bess_soc = bess.step(p_bess_actual)
            sc_soc = sc.step(p_sc_actual)
            
            # Calculate and log cost for this minute
            step_cost = (p_grid_actual * config.GRID_PRICE_TOU[k_sup] / 60) + \
                        (config.DG_COST_A * p_dg_actual**2 + config.DG_COST_B * p_dg_actual + (config.DG_COST_C if p_dg_actual > 0 else 0)) / 60
            cumulative_cost += step_cost if step_cost > 0 else 0
            
            # Log all data points for plotting
            history['t_in_c'].append(t_in)
            history['bess_soc'].append(bess_soc)
            history['sc_soc'].append(sc_soc)
            history['p_grid_kw'].append(p_grid_actual)
            history['p_dg_kw'].append(p_dg_actual)
            history['p_bess_kw'].append(p_bess_actual)
            history['p_sc_kw'].append(p_sc_actual)
            history['p_pv_kw'].append(pv_real)
            total_load_actual = load_real + p_hvac_plan[k_sup]
            history['p_load_kw'].append(total_load_actual)
            history['cost'].append(cumulative_cost)
            history['power_error_kw'].append(power_error_kw)
            history['p_bess_correction'].append(p_bess_correction)
            history['p_sc_correction'].append(p_sc_correction)
            history['p_bess_plan_interp'].append(p_bess_plan[k_sup])
            
    print(f"✅ Simulation complete in {time.time() - start_time:.2f} seconds.")
    
    # --- Calculate Baseline and Generate All Plots ---
    baseline_cost = calculate_baseline_cost(history)
    plot_primary_results(history, forecasts)
    plot_error_correction(history)
    plot_additional_visualizations(history, forecasts, p_bess_plan, p_hvac_plan, baseline_cost)
    
    plt.show()

def calculate_baseline_cost(history):
    """Calculates the operational cost without the MPC control system (Grid+PV only)."""
    print(" Bencmarking 'Without MPC' scenario...")
    baseline_cost_history = []
    cumulative_baseline_cost = 0
    
    for k in range(config.TOTAL_REALTIME_STEPS):
        k_sup = k // config.REALTIME_STEPS_PER_SUPERVISORY_STEP
        net_load = history['p_load_kw'][k] - history['p_pv_kw'][k]
        p_grid_baseline = max(0, net_load)
        step_cost = (p_grid_baseline * config.GRID_PRICE_TOU[k_sup]) / 60
        cumulative_baseline_cost += step_cost
        baseline_cost_history.append(cumulative_baseline_cost)
        
    return baseline_cost_history

def plot_primary_results(history, forecasts):
    """Generates the main plots to visualize the simulation outcome."""
    print("🎨 Generating primary results plots...")
    hours = np.arange(config.TOTAL_REALTIME_STEPS) / 60.0
    
    fig1, ax1 = plt.subplots(figsize=(15, 7))
    ax1.plot(hours, history['t_in_c'], 'b-', lw=2, label='Indoor Temp (°C)')
    ax1.axhline(config.T_ZONE_MIN_C, color='k', linestyle='--', label='Comfort Band')
    ax1.axhline(config.T_ZONE_MAX_C, color='k', linestyle='--')
    ax1.set(xlabel='Hour of Day', ylabel='Temperature (°C)', )
    ax1.grid(True)
    ax2 = ax1.twinx()
    forecast_hours = np.arange(config.SUPERVISORY_STEPS) * config.SUPERVISORY_TIMESTEP_MIN / 60.0
    ax2.plot(forecast_hours, forecasts['t_out_c'], 'r:', label='Outdoor Temp (°C)')
    ax2.set_ylabel('Outdoor Temp (°C)', color='r')
    fig1.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    ax3.stackplot(hours, history['p_grid_kw'], history['p_pv_kw'], history['p_dg_kw'], 
                  np.maximum(0, history['p_bess_kw']), np.maximum(0, history['p_sc_kw']),
                  labels=['Grid', 'PV', 'DG', 'BESS Discharge', 'SC Discharge'],
                  colors=['gray', 'orange', 'red', 'green', 'cyan'])
    ax3.plot(hours, history['p_load_kw'], 'k--', label='Total Load')
    ax3.set(ylabel='Power (kW)')
    ax3.legend(loc='upper left')
    ax3.grid(True)
    ax4.plot(hours, np.array(history['bess_soc']) * 100, 'm-', label='BESS SoC (%)')
    ax4.plot(hours, np.array(history['sc_soc']) * 100, 'c-', label='Supercapacitor SoC (%)')
    ax4.set(xlabel='Hour of Day', ylabel='State of Charge (%)', ylim=(0, 105))
    ax4.grid(True)
    ax4.legend()
    plt.tight_layout()

def plot_error_correction(history):
    """Generates a plot to visualize the Level 2 MPC error correction."""
    print("🎨 Generating error correction plot...")
    hours = np.arange(config.TOTAL_REALTIME_STEPS) / 60.0
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(hours, history['power_error_kw'], 'r-', lw=2, label='Power Error (Plan vs. Actual)')
    sc_correction = -np.array(history['p_sc_correction'])
    bess_correction = -np.array(history['p_bess_correction'])
    total_correction = sc_correction + bess_correction
    ax.plot(hours, total_correction, 'k--', lw=2, label='Total HESS Correction (BESS+SC)')
    ax.fill_between(hours, sc_correction, color='cyan', alpha=0.6, label='SC Correction')
    ax.fill_between(hours, sc_correction, total_correction, color='lightgreen', alpha=0.7, label='BESS Correction')
    ax.set(xlabel='Hour of Day', ylabel='Power (kW)')
    ax.grid(True)
    ax.legend()
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()

def plot_additional_visualizations(history, forecasts, p_bess_plan, p_hvac_plan, baseline_cost):
    """Generates a set of additional plots for deeper analysis."""
    print("🎨 Generating additional diagnostic plots...")
    hours = np.arange(config.TOTAL_REALTIME_STEPS) / 60.0
    forecast_hours = np.arange(config.SUPERVISORY_STEPS) * config.REALTIME_STEPS_PER_SUPERVISORY_STEP / 60.0

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(hours, history['cost'], 'r-', lw=2, label='With Fuzzy-Pid Control')
    ax1.plot(hours, baseline_cost, 'b--', lw=2, label='Without Fuzzy-Pid (Grid + PV Only)')
    ax1.set(xlabel='Hour of Day', ylabel='Cumulative Cost ($)')
    ax1.legend()
    ax1.grid(True)
    
    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax2.plot(forecast_hours, forecasts['pv_kw'], 'k--', label='Forecasted PV')
    ax2.plot(hours, history['p_pv_kw'], 'orange', alpha=0.8, label='Actual PV')
    ax2.set(ylabel='PV Power (kW)')
    ax2.legend()
    ax2.grid(True)
    load_forecast_interp = np.repeat(forecasts['uncontrollable_load_kw'] + p_hvac_plan, config.REALTIME_STEPS_PER_SUPERVISORY_STEP)
    ax3.plot(hours, load_forecast_interp, 'k--', label='Forecasted Total Load')
    ax3.plot(hours, history['p_load_kw'], 'b-', alpha=0.8, label='Actual Total Load')
    ax3.set(xlabel='Hour of Day', ylabel='Load Power (kW)')
    ax3.legend()
    ax3.grid(True)

    fig3, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(hours, history['p_bess_plan_interp'], 'r--', label='Planned BESS Dispatch (Level 1)')
    ax4.plot(hours, history['p_bess_kw'], 'g-', alpha=0.8, label='Actual BESS Dispatch (Level 2)')
    ax4.set(xlabel='Hour of Day', ylabel='BESS Power (kW)')
    ax4.legend()
    ax4.grid(True)

    fig4, ax5 = plt.subplots(figsize=(12, 6))
    ax5.plot(hours, history['p_bess_kw'], 'm-', label='BESS Power Profile')
    ax5.plot(hours, history['p_sc_kw'], 'c-', alpha=0.7, label='Supercapacitor Power Profile')
    ax5.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax5.set(xlabel='Hour of Day', ylabel='Power (kW)')
    ax5.legend()
    ax5.grid(True)
    
    plt.tight_layout()

if __name__ == '__main__':
    run_simulation_ss()

