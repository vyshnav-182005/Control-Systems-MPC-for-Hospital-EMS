import numpy as np
import sys
import os

# Adjust path to include the hospital_fuzzypid directory if run standalone
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hospital_fuzzypid"

from hospital_fuzzypid import config
from hospital_fuzzypid.models import ThermalZoneModel

class FuzzySupervisory:
    """Fuzzy logic controller for supervisory level to plan HVAC and power dispatch."""
    def __init__(self, forecasts):
        self.forecasts = forecasts
        self.steps = config.SUPERVISORY_STEPS
        self.thermal_zone = ThermalZoneModel(initial_temp_c=23.5)
        self.integral = 0.0  # For PID-like q_hvac
        self.previous_error = 0.0
        self.dt = config.SUPERVISORY_TIMESTEP_MIN / 60.0

    def membership(self, x, low, mid, high):
        """Triangular membership function for fuzzy sets: low, medium, high."""
        if x <= low:
            return 1.0, 0.0, 0.0
        elif x <= mid:
            low_m = (mid - x) / (mid - low)
            med_m = (x - low) / (mid - low)
            return low_m, med_m, 0.0
        elif x <= high:
            med_m = (high - x) / (high - mid)
            high_m = (x - mid) / (high - mid)
            return 0.0, med_m, high_m
        else:
            return 0.0, 0.0, 1.0

    def create_optimal_plan(self):
        plan = np.zeros((self.steps, 4))  # [p_grid, p_dg, p_bess, q_hvac]
        t_in = 23.5  # Initial indoor temp
        
        for k in range(self.steps):
            pv = self.forecasts['pv_kw'][k]
            load = self.forecasts['uncontrollable_load_kw'][k]
            t_out = self.forecasts['t_out_c'][k]
            internal_gain = self.forecasts['internal_gain_kw'][k]
            price = config.GRID_PRICE_TOU[k]
            
            # Fuzzy logic for q_hvac with PID-like adjustment
            temp_error = 23.5 - t_in
            neg_te, zero_te, pos_te = self.membership(temp_error, -5, 0, 5)
            self.integral += temp_error * self.dt
            derivative = (temp_error - self.previous_error) / self.dt
            self.previous_error = temp_error
            fuzzy_q = (-100 * neg_te + 0 * zero_te + 100 * pos_te) / (neg_te + zero_te + pos_te + 1e-6)
            pid_q = 15.0 * temp_error + 1.0 * self.integral + 0.2 * derivative  # Tuned gains
            q_hvac = 0.5 * fuzzy_q + 0.5 * pid_q  # Equal weight to fuzzy and PID
            q_hvac = np.clip(q_hvac, -100, 100)
            
            t_in = self.thermal_zone.step(t_out, internal_gain, q_hvac)
            p_hvac = q_hvac / config.HVAC_COP
            
            # Fuzzy logic for power dispatch
            net_load = load + p_hvac - pv
            low_nl, med_nl, high_nl = self.membership(net_load, -100, 0, 100)  # Wider range
            low_p, med_p, high_p = self.membership(price, 0.08, 0.15, 0.25)
            
            bess_charge = min(low_nl * low_p, low_nl * med_p)
            bess_discharge = min(high_nl * high_p, med_nl * high_p)
            p_bess = (-config.BESS_MAX_POWER_KW * bess_charge + config.BESS_MAX_POWER_KW * bess_discharge) / (bess_charge + bess_discharge + 1e-6)
            p_bess = np.clip(p_bess, -config.BESS_MAX_POWER_KW, config.BESS_MAX_POWER_KW)
            
            remaining_load = net_load - p_bess
            dg_on = min(high_nl * high_p, med_nl * high_p)
            p_dg = config.DG_MAX_POWER_KW * dg_on / (dg_on + 1e-6)
            p_dg = np.clip(p_dg, 0, config.DG_MAX_POWER_KW)
            
            p_grid = max(remaining_load - p_dg, 0)
            plan[k] = [p_grid, p_dg, p_bess, q_hvac]
        
        print("✅ Fuzzy supervisory plan created!")
        return plan

class PIDRealTime:
    """PID controller for real-time disturbance rejection."""
    def __init__(self, Kp=1.0, Ki=0.1, Kd=0.05):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.previous_error = 0.0
        self.dt = config.REALTIME_TIMESTEP_MIN / 60.0

    def dispatch(self, power_error_kw):
        self.integral += power_error_kw * self.dt
        derivative = (power_error_kw - self.previous_error) / self.dt
        self.previous_error = power_error_kw
        pid_output = self.Kp * power_error_kw + self.Ki * self.integral + self.Kd * derivative
        p_sc = np.clip(pid_output, -config.SC_MAX_POWER_KW, config.SC_MAX_POWER_KW)
        remaining_error = power_error_kw - p_sc
        p_bess = np.clip(remaining_error, -config.BESS_MAX_POWER_KW, config.BESS_MAX_POWER_KW)
        p_grid_correction = remaining_error - p_bess
        return p_bess, p_sc, p_grid_correction

if __name__ == "__main__":
    # For testing purposes, this can be run standalone with dummy data
    class DummyForecasts:
        def __init__(self):
            self.pv_kw = np.zeros(config.SUPERVISORY_STEPS)
            self.uncontrollable_load_kw = np.zeros(config.SUPERVISORY_STEPS)
            self.t_out_c = np.full(config.SUPERVISORY_STEPS, 20.0)
            self.internal_gain_kw = np.zeros(config.SUPERVISORY_STEPS)
    
    forecasts = DummyForecasts()
    sup = FuzzySupervisory(forecasts)
    plan = sup.create_optimal_plan()
    print("Plan shape:", plan.shape)
