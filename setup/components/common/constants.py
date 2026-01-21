"""Common utility constants.

These include Aurora 0.25 pre-trained model variable mappings and atmospheric levels.
"""

SURF_VAR_MAP = {
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
    "2m_temperature": "2t",
    "mean_sea_level_pressure": "msl",
}
STATIC_VAR_MAP = {
    "land_sea_mask": "lsm",
    "soil_type": "slt",
    "geopotential_at_surface": "z",
}
ATMOS_VAR_MAP = {
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "geopotential": "z",
}
ATMOS_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
