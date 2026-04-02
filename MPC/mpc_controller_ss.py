# hospital_mpc/mpc_controller_ss.py

import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from . import config

class SupervisoryMPC_SS:
    """Implements the Level 1 Supervisory MPC for 24-hour economic planning
       using a State-Space model formulation."""
    def __init__(self, forecasts):
        self.forecasts = forecasts
        self.steps = config.SUPERVISORY_STEPS  # Horizon N
        self.dt_hours = config.SUPERVISORY_TIMESTEP_MIN / 60.0

        # --- 1. State-space model definition ---
        # States: x = [T_in (°C), SOC_bess]
        # Controls: u = [P_grid (kW), P_dg (kW), P_bess (kW), Q_hvac (kW)]
        # Disturbances: d = [T_out (°C), Q_internal (kW)]
        self.num_states = 2
        self.num_controls = 4

        # Get system parameters from config
        C = config.C_MASS_KWH_PER_C
        R = config.R_WALL_C_PER_KW
        E_cap = config.BESS_CAPACITY_KWH

        # State matrix A: Describes how states evolve on their own
        self.A = np.array([
            [1 - self.dt_hours / (R * C), 0],
            [0, 1]
        ])
        
        # Input matrix B: Describes how controls affect the states
        self.B = np.array([
            [0, 0, 0, self.dt_hours / C],
            [0, 0, -self.dt_hours / E_cap, 0]
        ])

        # Disturbance matrix Dd: Describes how external disturbances affect states
        self.Dd = np.array([
            [self.dt_hours / (R * C), self.dt_hours / C], # T_out, Q_internal
            [0, 0]
        ])
        
        # Prepare the disturbance vector for the entire horizon
        self.disturbances = np.vstack([
            self.forecasts['t_out_c'],
            self.forecasts['internal_gain_kw']
        ]).T  # Shape (N, 2)


    def _objective_function(self, optim_vars):
        """Minimizes total operational cost.
        This function is now simpler as it only calculates cost and does not simulate."""
        # Unpack the control and state trajectories from the single optimization vector
        U = optim_vars[:self.steps * self.num_controls].reshape((self.steps, self.num_controls))
        X = optim_vars[self.steps * self.num_controls:].reshape((self.steps, self.num_states))
        
        p_grid, p_dg, p_bess, q_hvac = U.T
        t_in_trajectory = X[:, 0]  # First state is T_in

        # --- Cost Calculations (identical to the previous version) ---
        grid_cost = np.sum(config.GRID_PRICE_TOU * p_grid)

        dg_on = p_dg > 1e-3
        dg_cost = np.sum(
            config.DG_COST_A * p_dg[dg_on]**2 +
            config.DG_COST_B * p_dg[dg_on] +
            config.DG_COST_C
        )

        discharged_energy = np.sum(np.maximum(0, p_bess)) * self.dt_hours
        batt_deg_cost = discharged_energy * config.BATT_DEG_COST_PER_KWH
        
        temp_violations = np.maximum(0, config.T_ZONE_MIN_C - t_in_trajectory) + \
                          np.maximum(0, t_in_trajectory - config.T_ZONE_MAX_C)
        thermal_penalty = np.sum(temp_violations) * config.THERMAL_PENALTY_WEIGHT

        return grid_cost + dg_cost + batt_deg_cost + thermal_penalty


    def _constraints(self, optim_vars):
        """Defines all equality constraints: system dynamics and power balance."""
        # Unpack variables
        U = optim_vars[:self.steps * self.num_controls].reshape((self.steps, self.num_controls))
        X = optim_vars[self.steps * self.num_controls:].reshape((self.steps, self.num_states))

        # Initial state (a known constant, not a decision variable)
        x0 = np.array([23.5, config.BESS_INITIAL_SOC])  # T_in_initial, SOC_bess_initial
        
        # Prepend initial state to the state trajectory for easier calculations
        X_full = np.vstack([x0, X])

        # --- 1. System Dynamics Constraints ---
        # Enforce x_{k+1} = Ax_k + Bu_k + Dd_k for the entire horizon
        dynamics_residuals = []
        for k in range(self.steps):
            x_k = X_full[k]
            u_k = U[k]
            d_k = self.disturbances[k]
            x_k_plus_1_predicted = self.A @ x_k + self.B @ u_k + self.Dd @ d_k
            
            # Constraint: x_{k+1} - predicted_x_{k+1} must equal 0
            dynamics_residuals.extend(X_full[k+1] - x_k_plus_1_predicted)
        
        # --- 2. Power Balance Constraints ---
        p_grid, p_dg, p_bess, q_hvac = U.T
        p_hvac = q_hvac / config.HVAC_COP
        
        power_balance_residuals = (self.forecasts['pv_kw'] + p_grid + p_dg + p_bess -
                                   self.forecasts['uncontrollable_load_kw'] - p_hvac)

        # The optimizer will force both residuals to be zero
        return np.concatenate([dynamics_residuals, power_balance_residuals])

    def create_optimal_plan(self):
        """Runs the optimization to find the 24-hour energy schedule."""
        # The optimization vector contains the entire control AND state trajectories
        len_U = self.steps * self.num_controls
        len_X = self.steps * self.num_states
        
        # --- Bounds for control inputs U ---
        lower_u = [0, 0, -config.BESS_MAX_POWER_KW, -100] * self.steps
        upper_u = [np.inf, config.DG_MAX_POWER_KW, config.BESS_MAX_POWER_KW, 100] * self.steps

        # --- Bounds for states X ---
        # T_in is unconstrained here (handled by penalty), SoC is [0, 1]
        lower_x = [-np.inf, 0.0] * self.steps
        upper_x = [np.inf, 1.0] * self.steps
        
        bounds = Bounds(
            np.concatenate([lower_u, lower_x]),
            np.concatenate([upper_u, upper_x])
        )

        # --- All constraints must equal zero ---
        total_constraints = self.steps * self.num_states + self.steps
        constraints = NonlinearConstraint(
            fun=self._constraints,
            lb=np.zeros(total_constraints),
            ub=np.zeros(total_constraints)
        )
        
        # Initial guess (all zeros)
        x0_optim = np.zeros(len_U + len_X)

        print("🚀 Starting Level 1 Supervisory MPC (State-Space) optimization...")
        result = minimize(self._objective_function, x0_optim, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'disp': True, 'maxiter': 300, 'ftol': 1e-4})

        if result.success:
            print("✅ Level 1 (State-Space) Optimization successful!")
            # Extract the optimal control plan U and return it
            U_optimal = result.x[:len_U].reshape((self.steps, self.num_controls))
            return U_optimal
        else:
            print(f"⚠️ Level 1 (State-Space) Optimization failed: {result.message}")
            return None

class RealTimeMPC:
    """Implements the Level 2 Real-Time MPC for tracking and disturbance rejection.
       This class is unchanged as its logic is independent of the supervisory formulation."""
    def dispatch(self, power_error_kw):
        p_sc = np.clip(power_error_kw, -config.SC_MAX_POWER_KW, config.SC_MAX_POWER_KW)
        remaining_error = power_error_kw - p_sc
        p_bess = np.clip(remaining_error, -config.BESS_MAX_POWER_KW, config.BESS_MAX_POWER_KW)
        p_grid_correction = remaining_error - p_bess
        return p_bess, p_sc, p_grid_correction