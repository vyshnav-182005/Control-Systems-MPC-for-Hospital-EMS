# hospital_mpc/models.py

from . import config
import numpy as np

class ThermalZoneModel:
    """Implements the RC thermal model for a hospital zone."""
    def __init__(self, initial_temp_c):
        self.t_in = initial_temp_c
        self.C = config.C_MASS_KWH_PER_C
        self.R = config.R_WALL_C_PER_KW

    def step(self, t_out_c, q_internal_kw, q_hvac_kw):
        """Updates indoor temperature over one REALTIME_TIMESTEP_MIN."""
        dt_hours = config.REALTIME_TIMESTEP_MIN / 60.0
        
        dT_in = (
            (t_out_c - self.t_in) / self.R +
            q_internal_kw +
            q_hvac_kw
        ) * (dt_hours / self.C)
        
        self.t_in += dT_in
        return self.t_in

class BESSModel:
    """Models the Battery Energy Storage System."""
    def __init__(self):
        self.soc = config.BESS_INITIAL_SOC
        self.capacity_kwh = config.BESS_CAPACITY_KWH
        self.efficiency = config.BESS_EFFICIENCY

    def step(self, power_kw):
        """Updates state of charge over one REALTIME_TIMESTEP_MIN."""
        dt_hours = config.REALTIME_TIMESTEP_MIN / 60.0
        
        if power_kw > 0: # Discharging
            energy_change_kwh = power_kw * dt_hours / self.efficiency
        else: # Charging
            energy_change_kwh = power_kw * dt_hours * self.efficiency
            
        self.soc -= energy_change_kwh / self.capacity_kwh
        self.soc = np.clip(self.soc, 0, 1)
        return self.soc

class SCModel:
    """Models the Supercapacitor."""
    def __init__(self):
        self.soc = config.SC_INITIAL_SOC
        self.capacity_kwh = config.SC_CAPACITY_KWH
        self.efficiency = config.SC_EFFICIENCY

    def step(self, power_kw):
        """Updates state of charge over one REALTIME_TIMESTEP_MIN."""
        dt_hours = config.REALTIME_TIMESTEP_MIN / 60.0

        if power_kw > 0: # Discharging
            energy_change_kwh = power_kw * dt_hours / self.efficiency
        else: # Charging
            energy_change_kwh = power_kw * dt_hours * self.efficiency

        self.soc -= energy_change_kwh / self.capacity_kwh
        self.soc = np.clip(self.soc, 0, 1)
        return self.soc
