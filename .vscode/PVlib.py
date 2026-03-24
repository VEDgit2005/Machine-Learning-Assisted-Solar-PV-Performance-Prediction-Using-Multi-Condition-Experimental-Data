import pandas as pd
import pvlib
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

# ---------------------------
# 1. LOCATION (SRM KTR, Chennai)
# ---------------------------
latitude = 12.8230
longitude = 80.0444
tz = 'Asia/Kolkata'

location = pvlib.location.Location(latitude, longitude, tz=tz)

# ---------------------------
# 2. TIME RANGE
# ---------------------------
# You can change year or frequency
times = pd.date_range(
    start='2023-01-01',
    end='2023-12-31 23:59:00',
    freq='1h',   # ✅ FIXED
    tz=tz
)

# ---------------------------
# 3. SOLAR POSITION
# ---------------------------
solar_position = location.get_solarposition(times)

# ---------------------------
# 4. CLEAR SKY IRRADIANCE
# ---------------------------
clearsky = location.get_clearsky(times, model='ineichen')

# ---------------------------
# 5. AIR MASS
# ---------------------------
airmass = pvlib.atmosphere.get_relative_airmass(solar_position['zenith'])

# ---------------------------
# 6. PANEL CONFIG
# ---------------------------
surface_tilt = 13   # approx latitude
surface_azimuth = 180  # south-facing

# ---------------------------
# 7. POA IRRADIANCE
# ---------------------------
poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=surface_tilt,
    surface_azimuth=surface_azimuth,
    dni=clearsky['dni'],
    ghi=clearsky['ghi'],
    dhi=clearsky['dhi'],
    solar_zenith=solar_position['zenith'],
    solar_azimuth=solar_position['azimuth']
)

# ---------------------------
# 8. TEMPERATURE MODEL
# ---------------------------
temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

cell_temperature = pvlib.temperature.sapm_cell(
    poa_global=poa['poa_global'],
    temp_air=30,   # constant for now
    wind_speed=1,  # constant
    **temp_params
)

# ---------------------------
# 9. PV POWER (PVWatts)
# ---------------------------
pdc0 = 250  # panel rating (W)
gamma_pdc = -0.004  # temp coefficient

dc_power = pvlib.pvsystem.pvwatts_dc(
    g_poa_effective=poa['poa_global'],  # ✅ FIXED NAME
    temp_cell=cell_temperature,
    pdc0=pdc0,
    gamma_pdc=gamma_pdc
)


# ---------------------------
# 10. FINAL DATASET
# ---------------------------
df = pd.DataFrame({
    'datetime': times,
    'zenith': solar_position['zenith'],
    'elevation': solar_position['elevation'],
    'azimuth': solar_position['azimuth'],
    'ghi': clearsky['ghi'],
    'dni': clearsky['dni'],
    'dhi': clearsky['dhi'],
    'poa_global': poa['poa_global'],
    'airmass': airmass,
    'cell_temperature': cell_temperature,
    'dc_power': dc_power
})

# ---------------------------
# 11. CLEANING (important)
# ---------------------------
# Remove night noise (optional)
df['dc_power'] = df['dc_power'].clip(lower=0)

# ---------------------------
# 12. SAVE
# ---------------------------
df.to_csv("pvlib_chennai_srm_dataset.csv", index=False)

print("✅ Dataset created successfully!")
print(df.head())