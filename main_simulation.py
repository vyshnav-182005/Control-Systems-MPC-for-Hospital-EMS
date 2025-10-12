# hospital_mpc/main_simulation.py

import matplotlib.pyplot as plt
import numpy as np
import time

from . import config
from .forecasting import get_ann_forecasts, get_real_time_measurements
from .models import ThermalZoneModel, BESSModel, SCModel
from .mpc_controller import SupervisoryMPC, RealTimeMPC

def run_simulation():
    """Main function to run the hierarchical electro-thermal MPC simulation."""
    start_time = time.time()
    
    # 1. Get 24-hour Forecasts using the ANN
    forecasts = get_ann_forecasts()
    
    # 2. Run Level 1 Supervisory MPC to get the optimal 24-hour plan
    sup_mpc = SupervisoryMPC(forecasts)
    optimal_plan = sup_mpc.create_optimal_plan()
    
    if optimal_plan is None:
        return

    p_grid_plan, p_dg_plan, p_bess_plan, q_hvac_plan = optimal_plan.T
    p_hvac_plan = q_hvac_plan / config.HVAC_COP
    
    # 3. Initialize Level 2 MPC and system models
    rt_mpc = RealTimeMPC()
    thermal_zone = ThermalZoneModel(initial_temp_c=23.5)
    bess = BESSModel()
    sc = SCModel()

    # History for plotting at 1-minute resolution
    history = {
        't_in_c': [], 'bess_soc': [], 'sc_soc': [], 'p_grid_kw': [], 
        'p_dg_kw': [], 'p_bess_kw': [], 'p_sc_kw': [], 'p_pv_kw': [], 
        'p_load_kw': [], 'cost': []
    }

    print("\n⚡️ Starting Level 2 real-time simulation loop (1-min steps)...")
    cumulative_cost = 0
    
    # Create the time axis for interpolation once before the loop
    forecast_minutes = np.arange(config.SUPERVISORY_STEPS) * config.SUPERVISORY_TIMESTEP_MIN
    
    for k_sup in range(config.SUPERVISORY_STEPS):
        for k_rt in range(config.REALTIME_STEPS_PER_SUPERVISORY_STEP):
            k_total = k_sup * config.REALTIME_STEPS_PER_SUPERVISORY_STEP + k_rt
            
            # Get "real" measurements for the current minute
            real_measurements = get_real_time_measurements(forecasts, k_sup, k_rt)
            
            # --- Level 2: Real-Time Control Logic ---
            # Determine the error between planned and actual power needs
            planned_net_load = forecasts['uncontrollable_load_kw'][k_sup] + p_hvac_plan[k_sup] - forecasts['pv_kw'][k_sup]
            real_net_load = real_measurements['uncontrollable_load_kw'] + p_hvac_plan[k_sup] - real_measurements['pv_kw']
            power_error_kw = real_net_load - planned_net_load
            
            # Use RT_MPC to dispatch HESS and grid to correct the error
            p_bess_correction, p_sc_correction, p_grid_correction = rt_mpc.dispatch(power_error_kw)
            
            # Calculate actual power values by adding corrections to the plan
            p_bess_actual = p_bess_plan[k_sup] + p_bess_correction
            p_sc_actual = p_sc_correction
            p_grid_actual = p_grid_plan[k_sup] + p_grid_correction
            p_dg_actual = p_dg_plan[k_sup]
            
            # --- Update System State ---
            # Interpolate forecast values to the current 1-minute step
            t_out_interp = np.interp(k_total, forecast_minutes, forecasts['t_out_c'])
            # --- FIX: Added interpolation for internal_gain for consistency ---
            internal_gain_interp = np.interp(k_total, forecast_minutes, forecasts['internal_gain_kw'])

            t_in = thermal_zone.step(t_out_interp, internal_gain_interp, q_hvac_plan[k_sup])
            bess_soc = bess.step(p_bess_actual)
            sc_soc = sc.step(p_sc_actual)
            
            # --- Record Data ---
            step_cost = (p_grid_actual * config.GRID_PRICE_TOU[k_sup] / 60) + \
                        (config.DG_COST_A * p_dg_actual**2 + config.DG_COST_B * p_dg_actual + (config.DG_COST_C if p_dg_actual > 0 else 0)) / 60
            cumulative_cost += step_cost
            
            history['t_in_c'].append(t_in)
            history['bess_soc'].append(bess_soc)
            history['sc_soc'].append(sc_soc)
            history['p_grid_kw'].append(p_grid_actual)
            history['p_dg_kw'].append(p_dg_actual)
            history['p_bess_kw'].append(p_bess_actual)
            history['p_sc_kw'].append(p_sc_actual)
            history['p_pv_kw'].append(real_measurements['pv_kw'])
            total_load = real_measurements['uncontrollable_load_kw'] + p_hvac_plan[k_sup]
            history['p_load_kw'].append(total_load)
            history['cost'].append(cumulative_cost)
            
    print(f"✅ Simulation complete in {time.time() - start_time:.2f} seconds.")
    plot_results(history, forecasts)

def plot_results(history, forecasts):
    """Generates plots to visualize the simulation outcome."""
    minutes = np.arange(config.TOTAL_REALTIME_STEPS)
    hours = minutes / 60.0
    
    # Fig 1: Thermal Performance
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax1.plot(hours, history['t_in_c'], 'b-', lw=2, label='Indoor Temp (°C)')
    ax1.axhline(config.T_ZONE_MIN_C, color='k', linestyle='--', label='Comfort Band')
    ax1.axhline(config.T_ZONE_MAX_C, color='k', linestyle='--')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Temperature (°C)', color='b')
    ax1.set_title('Thermal Performance with 1-Minute Resolution')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    forecast_hours = np.arange(config.SUPERVISORY_STEPS) * config.SUPERVISORY_TIMESTEP_MIN / 60.0
    ax2.plot(forecast_hours, forecasts['t_out_c'], 'r:', label='Outdoor Temp (°C)')
    ax2.set_ylabel('Outdoor Temp (°C)', color='r')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    
    # Fig 2 & 3: Electrical Dispatch and HESS SoC
    fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    ax3.stackplot(hours, history['p_grid_kw'], history['p_pv_kw'], history['p_dg_kw'], 
                  np.maximum(0, history['p_bess_kw']), np.maximum(0, history['p_sc_kw']),
                  labels=['Grid', 'PV', 'DG', 'BESS Discharge', 'SC Discharge'],
                  colors=['gray', 'orange', 'red', 'green', 'cyan'])
    ax3.plot(hours, history['p_load_kw'], 'k--', label='Total Load')
    ax3.set_ylabel('Power (kW)')
    ax3.set_title('Optimal Power Dispatch (1-Minute Resolution)')
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    ax4.plot(hours, np.array(history['bess_soc']) * 100, 'm-', label='BESS SoC (%)')
    ax4.plot(hours, np.array(history['sc_soc']) * 100, 'c-', label='Supercapacitor SoC (%)')
    ax4.set_ylabel('State of Charge (%)')
    ax4.set_xlabel('Hour of Day')
    ax4.set_title('HESS Operation: BESS for Arbitrage, SC for Fluctuations')
    ax4.grid(True)
    ax4.set_ylim(0, 105)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_simulation()