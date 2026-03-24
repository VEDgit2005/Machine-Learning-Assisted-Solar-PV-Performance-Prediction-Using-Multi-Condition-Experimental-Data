import pandas as pd
import pvlib
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import numpy as np

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
times = pd.date_range(
    start='2023-01-01',
    end='2023-12-31 23:00:00',
    freq='1h',
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
# 6. FIXED PARAMETERS
# ---------------------------
surface_azimuth = 180  # south-facing
temp_air = 30
wind_speed = 1

pdc0 = 250
gamma_pdc = -0.004

temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

# ---------------------------
# 7. TILT RANGE (KEY PART 🔥)
# ---------------------------
tilt_angles = np.arange(0, 41, 2)  # 0° to 40° (step 2°)

# Store results
results = []

# ---------------------------
# 8. LOOP OVER EACH TIME
# ---------------------------
for i, time in enumerate(times):

    best_tilt = None
    max_power = -np.inf

    for tilt in tilt_angles:

        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=surface_azimuth,
            dni=clearsky['dni'].iloc[i],
            ghi=clearsky['ghi'].iloc[i],
            dhi=clearsky['dhi'].iloc[i],
            solar_zenith=solar_position['zenith'].iloc[i],
            solar_azimuth=solar_position['azimuth'].iloc[i]
        )

        cell_temp = pvlib.temperature.sapm_cell(
            poa_global=poa['poa_global'],
            temp_air=temp_air,
            wind_speed=wind_speed,
            **temp_params
        )

        power = pvlib.pvsystem.pvwatts_dc(
            effective_irradiance=poa['poa_global'],
            temp_cell=cell_temp,
            pdc0=pdc0,
            gamma_pdc=gamma_pdc
        )

        # Track best tilt
        if power > max_power:
            max_power = power
            best_tilt = tilt

    results.append({
        'datetime': time,
        'zenith': solar_position['zenith'].iloc[i],
        'azimuth': solar_position['azimuth'].iloc[i],
        'ghi': clearsky['ghi'].iloc[i],
        'dni': clearsky['dni'].iloc[i],
        'dhi': clearsky['dhi'].iloc[i],
        'airmass': airmass.iloc[i],
        'optimal_tilt': best_tilt,
        'max_power': max_power
    })

# ---------------------------
# 9. FINAL DATASET
# ---------------------------
df = pd.DataFrame(results)

# Add useful features
df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour

# ---------------------------
# 10. CLEANING
# ---------------------------
# Remove night (no solar info)
df = df[df['ghi'] > 0]

# ---------------------------
# 11. SAVE
# ---------------------------
df.to_csv("pvlib_optimal_tilt_dataset.csv", index=False)

print("✅ Optimal tilt dataset created!")
print(df.head())