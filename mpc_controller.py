# hospital_mpc/mpc_controller.py

import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from . import config

class SupervisoryMPC:
    """Implements the Level 1 Supervisory MPC for 24-hour economic planning."""
    def __init__(self, forecasts):
        self.forecasts = forecasts
        self.steps = config.SUPERVISORY_STEPS
        self.num_vars_per_step = 4  # [P_grid, P_dg, P_bess, Q_hvac]

    def _objective_function(self, x):
        """Minimizes total operational cost."""
        plan = x.reshape((self.steps, self.num_vars_per_step))
        p_grid, p_dg, p_bess, q_hvac = plan.T

        grid_cost = np.sum(config.GRID_PRICE_TOU * p_grid)
        
        # --- FIX: Vectorized DG cost calculation for efficiency ---
        # Calculate cost only for steps where the generator is on (p_dg > 0)
        dg_on = p_dg > 1e-3
        dg_cost = np.sum(
            config.DG_COST_A * p_dg[dg_on]**2 + 
            config.DG_COST_B * p_dg[dg_on] + 
            config.DG_COST_C
        )
        
        discharged_energy = np.sum(np.maximum(0, p_bess)) * (config.SUPERVISORY_TIMESTEP_MIN / 60.0)
        batt_deg_cost = discharged_energy * config.BATT_DEG_COST_PER_KWH

        # Simulate thermal trajectory
        t_in_trajectory = np.zeros(self.steps + 1)
        t_in_trajectory[0] = 23.0
        for k in range(self.steps):
            dt_hours = config.SUPERVISORY_TIMESTEP_MIN / 60.0
            dT_in = ((self.forecasts['t_out_c'][k] - t_in_trajectory[k]) / config.R_WALL_C_PER_KW +
                     self.forecasts['internal_gain_kw'][k] + q_hvac[k]) * (dt_hours / config.C_MASS_KWH_PER_C)
            t_in_trajectory[k+1] = t_in_trajectory[k] + dT_in
        
        temp_violations = np.maximum(0, config.T_ZONE_MIN_C - t_in_trajectory[1:]) + \
                          np.maximum(0, t_in_trajectory[1:] - config.T_ZONE_MAX_C)
        thermal_penalty = np.sum(temp_violations) * config.THERMAL_PENALTY_WEIGHT

        return grid_cost + dg_cost + batt_deg_cost + thermal_penalty

    def _constraints(self, x):
        """Defines the power balance equality constraint."""
        plan = x.reshape((self.steps, self.num_vars_per_step))
        p_grid, p_dg, p_bess, q_hvac = plan.T
        
        p_hvac = q_hvac / config.HVAC_COP
        power_balance = self.forecasts['pv_kw'] + p_grid + p_dg + p_bess - self.forecasts['uncontrollable_load_kw'] - p_hvac
        return power_balance
        
    def create_optimal_plan(self):
        """Runs the optimization to find the 24-hour energy schedule."""
        lower_bounds = [0, 0, -config.BESS_MAX_POWER_KW, -100] * self.steps
        upper_bounds = [np.inf, config.DG_MAX_POWER_KW, config.BESS_MAX_POWER_KW, 100] * self.steps
        bounds = Bounds(lower_bounds, upper_bounds)
        constraints = NonlinearConstraint(self._constraints, lb=0, ub=0)
        x0 = np.zeros(self.steps * self.num_vars_per_step)

        print("🚀 Starting Level 1 Supervisory MPC optimization...")
        result = minimize(self._objective_function, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'disp': True, 'maxiter': 300})
        
        if result.success:
            print("✅ Level 1 Optimization successful!")
            return result.x.reshape((self.steps, self.num_vars_per_step))
        else:
            print(f"⚠️ Level 1 Optimization failed: {result.message}")
            return None

class RealTimeMPC:
    """Implements the Level 2 Real-Time MPC for tracking and disturbance rejection."""
    def __init__(self):
        # This MPC is simplified to a rule-based controller for speed, matching the proposal's intent.
        pass

    def dispatch(self, power_error_kw):
        """Dispatches HESS to compensate for power errors."""
        # Supercapacitor handles high-frequency component of the error
        p_sc = np.clip(power_error_kw, -config.SC_MAX_POWER_KW, config.SC_MAX_POWER_KW)
        
        # BESS handles the remaining, slower-moving error
        remaining_error = power_error_kw - p_sc
        p_bess = np.clip(remaining_error, -config.BESS_MAX_POWER_KW, config.BESS_MAX_POWER_KW)
        
        # Grid takes up whatever is left
        p_grid_correction = remaining_error - p_bess
        
        return p_bess, p_sc, p_grid_correction