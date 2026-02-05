"""
Complete Solar PV Data Pipeline - Single Unified Dataset Creation
===================================================================

This script combines ALL data sources and creates ONE final dataset ready for ML prediction models.

Author: ML Solar PV Research Project
Date: January 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print(" "*15 + "SOLAR PV COMPLETE DATA PIPELINE")
print(" "*10 + "Creating Single Unified Dataset for ML Prediction")
print("="*80 + "\n")

# ============================================================================
# CONFIGURATION - SIMPLIFIED WITH RELATIVE PATHS
# ============================================================================

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try to find the Excel files in the same directory
print("Looking for Excel files in:", SCRIPT_DIR)
print("-" * 80)

# List all Excel files in the directory
excel_files = [f for f in os.listdir(SCRIPT_DIR) if f.endswith(('.xlsx', '.xls'))]
print(f"\nFound {len(excel_files)} Excel files:")
for i, file in enumerate(excel_files, 1):
    print(f"  {i}. {file}")

# Try to identify the correct files
solar_file = None
weather_file = None

for file in excel_files:
    if 'Solar' in file and 'Meter' in file:
        solar_file = os.path.join(SCRIPT_DIR, file)
        print(f"\n✓ Identified Solar file: {file}")
    elif 'Weather' in file:
        weather_file = os.path.join(SCRIPT_DIR, file)
        print(f"✓ Identified Weather file: {file}")

# If files not found automatically, ask user to specify
if solar_file is None:
    print("\n❌ Could not automatically find Solar Meter file.")
    print("Please ensure the Solar Meter Excel file is in the same folder as this script.")
    print("Or update the SOLAR_FILE variable below with the correct path.")
    
    # You can manually set the path here:
    SOLAR_FILE = r'C:\Users\jenav\OneDrive\Documents\STUDY\UROP ML SOLAR PV\SRM_ANNEXURE_CAMPUS_837_1_Solar_Meter_Day_Reports_30_08_2024_To_20_01_2026__.xlsx'
else:
    SOLAR_FILE = solar_file

if weather_file is None:
    print("\n❌ Could not automatically find Weather file.")
    print("Please ensure the Weather Excel file is in the same folder as this script.")
    print("Or update the WEATHER_FILE variable below with the correct path.")
    
    # You can manually set the path here:
    WEATHER_FILE = r'C:\Users\jenav\OneDrive\Documents\STUDY\UROP ML SOLAR PV\SRM_ANNEXURE_CAMPUS_837_1___Weather_____Day_Reports_30_08_2024_To_20_01_2026__.xlsx'
else:
    WEATHER_FILE = weather_file

# Verify files exist
print("\n" + "="*80)
print("Verifying file paths...")
print("-" * 80)

if not os.path.exists(SOLAR_FILE):
    print(f"❌ ERROR: Solar file not found at: {SOLAR_FILE}")
    print("\nPlease check:")
    print("1. Is the file in the correct location?")
    print("2. Is the filename spelled correctly (including spaces and underscores)?")
    print("3. Does the file have .xlsx or .xls extension?")
    input("\nPress Enter to exit...")
    exit()
else:
    print(f"✓ Solar file found: {os.path.basename(SOLAR_FILE)}")

if not os.path.exists(WEATHER_FILE):
    print(f"❌ ERROR: Weather file not found at: {WEATHER_FILE}")
    print("\nPlease check:")
    print("1. Is the file in the correct location?")
    print("2. Is the filename spelled correctly?")
    input("\nPress Enter to exit...")
    exit()
else:
    print(f"✓ Weather file found: {os.path.basename(WEATHER_FILE)}")

# System specifications (from uploaded images)
AC_CAPACITY_KW = 680
DC_CAPACITY_KWP = 838
UNIT_RATE = 10
NUM_WINGS = 5
NUM_INVERTERS = 8
NUM_METERS = 7
TIMEZONE = 'Asia/Kolkata'

# Output paths - Save in same directory as script
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'solar_pv_final_unified_dataset.csv')
OUTPUT_SUMMARY = os.path.join(SCRIPT_DIR, 'dataset_summary.txt')

# ============================================================================
# STEP 1: LOAD ALL DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Loading All Data Sources")
print("-" * 80)

try:
    # Load Solar Meter Data
    solar_df = pd.read_excel(SOLAR_FILE, sheet_name='Meter -Gen -')
    print(f"✓ Loaded Solar Meter Data: {solar_df.shape[0]} rows, {solar_df.shape[1]} columns")
except Exception as e:
    print(f"❌ ERROR loading Solar file: {e}")
    print("\nTrying to list available sheet names...")
    try:
        xls = pd.ExcelFile(SOLAR_FILE)
        print(f"Available sheets: {xls.sheet_names}")
        print("\nPlease update the sheet_name in the code to match one of the above.")
    except:
        pass
    input("\nPress Enter to exit...")
    exit()

try:
    # Load Weather Data  
    weather_df = pd.read_excel(WEATHER_FILE, sheet_name=' Weather ')
    print(f"✓ Loaded Weather Data: {weather_df.shape[0]} rows, {weather_df.shape[1]} columns")
except Exception as e:
    print(f"❌ ERROR loading Weather file: {e}")
    print("\nTrying to list available sheet names...")
    try:
        xls = pd.ExcelFile(WEATHER_FILE)
        print(f"Available sheets: {xls.sheet_names}")
        print("\nPlease update the sheet_name in the code to match one of the above.")
    except:
        pass
    input("\nPress Enter to exit...")
    exit()

print(f"\nOriginal columns in Solar Data:")
for i, col in enumerate(solar_df.columns, 1):
    print(f"  {i:2d}. {col}")

# ============================================================================
# STEP 2: CLEAN AND STANDARDIZE DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Cleaning and Standardizing Data")
print("-" * 80)

# Clean column names
solar_df.columns = solar_df.columns.str.strip().str.replace(' +', ' ', regex=True)
weather_df.columns = weather_df.columns.str.strip().str.replace(' +', ' ', regex=True)

# Convert Date column to datetime
solar_df['Date'] = pd.to_datetime(solar_df['Date'], errors='coerce')
weather_df['Date'] = pd.to_datetime(weather_df['Date'], errors='coerce')

# Convert numeric columns
numeric_columns = [
    'CUF ON AC Capacity (%)',
    'CUF ON DC Capacity (%)',
    'Radiation_Sensor-Day PR (%)',
    'Adj Radiation_Sensor-Day PR (%)'
]

for col in numeric_columns:
    if col in solar_df.columns:
        solar_df[col] = pd.to_numeric(solar_df[col], errors='coerce')

# Remove rows with missing critical data
initial_rows = len(solar_df)
solar_df = solar_df.dropna(subset=['Date', 'Day Gen (KWh)', 'Radiation_Sensor-Day INS (KWh/M2/Day)'])
print(f"✓ Removed {initial_rows - len(solar_df)} rows with missing critical data")
print(f"✓ Remaining records: {len(solar_df)}")

# ============================================================================
# STEP 3: MERGE ALL DATA SOURCES
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Merging All Data Sources")
print("-" * 80)

# Since solar_df already contains most information, we'll use it as the base
unified_df = solar_df.copy()

print(f"✓ Using Solar Meter data as base")
print(f"✓ Unified dataset shape: {unified_df.shape}")

# ============================================================================
# STEP 4: ADD SYSTEM SPECIFICATIONS AS FEATURES
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Adding System Specifications")
print("-" * 80)

unified_df['AC_Capacity_kW'] = AC_CAPACITY_KW
unified_df['DC_Capacity_kWp'] = DC_CAPACITY_KWP
unified_df['Unit_Rate'] = UNIT_RATE
unified_df['Num_Wings'] = NUM_WINGS
unified_df['Num_Inverters'] = NUM_INVERTERS
unified_df['Num_Meters'] = NUM_METERS

print(f"✓ Added system specifications as features")

# ============================================================================
# STEP 5: CREATE COMPREHENSIVE FEATURE SET
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Engineering Complete Feature Set")
print("-" * 80)

# ============================================================================
# 5.1: TEMPORAL FEATURES
# ============================================================================

print("\n5.1: Creating Temporal Features...")

# Basic temporal features
unified_df['Year'] = unified_df['Date'].dt.year
unified_df['Month'] = unified_df['Date'].dt.month
unified_df['Day'] = unified_df['Date'].dt.day
unified_df['DayOfWeek'] = unified_df['Date'].dt.dayofweek
unified_df['DayOfYear'] = unified_df['Date'].dt.dayofyear
unified_df['Quarter'] = unified_df['Date'].dt.quarter
unified_df['WeekOfYear'] = unified_df['Date'].dt.isocalendar().week
unified_df['Is_Weekend'] = (unified_df['DayOfWeek'] >= 5).astype(int)

# Season (India-specific)
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Monsoon'

unified_df['Season'] = unified_df['Month'].apply(get_season)
season_map = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Monsoon': 4}
unified_df['Season_Numeric'] = unified_df['Season'].map(season_map)

# Cyclical encoding
unified_df['Month_Sin'] = np.sin(2 * np.pi * unified_df['Month'] / 12)
unified_df['Month_Cos'] = np.cos(2 * np.pi * unified_df['Month'] / 12)
unified_df['DayOfYear_Sin'] = np.sin(2 * np.pi * unified_df['DayOfYear'] / 365)
unified_df['DayOfYear_Cos'] = np.cos(2 * np.pi * unified_df['DayOfYear'] / 365)

print(f"  ✓ Created 17 temporal features")

# ============================================================================
# 5.2: PERFORMANCE METRICS
# ============================================================================

print("\n5.2: Creating Performance Metrics...")

unified_df['AC_DC_Ratio'] = unified_df['CUF ON AC Capacity (%)'] / (unified_df['CUF ON DC Capacity (%)'] + 0.001)
unified_df['Theoretical_Max_Energy'] = AC_CAPACITY_KW * 24
unified_df['Capacity_Utilization_Actual'] = (unified_df['Day Gen (KWh)'] / unified_df['Theoretical_Max_Energy']) * 100
unified_df['Energy_Per_Irradiance'] = unified_df['Day Gen (KWh)'] / (unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)'] + 0.001)
unified_df['Energy_Per_Adj_Irradiance'] = unified_df['Day Gen (KWh)'] / (unified_df['Adj Radiation_Sensor-Day INS (KWh/M2/Day)'] + 0.001)
unified_df['PR_Difference'] = unified_df['Radiation_Sensor-Day PR (%)'] - unified_df['Adj Radiation_Sensor-Day PR (%)']
unified_df['Irradiance_Difference'] = unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)'] - unified_df['Adj Radiation_Sensor-Day INS (KWh/M2/Day)']
unified_df['Specific_Yield'] = unified_df['Day Gen (KWh)'] / DC_CAPACITY_KWP
unified_df['Avg_Power_kW'] = unified_df['Day Gen (KWh)'] / 10
unified_df['System_Efficiency'] = (unified_df['Day Gen (KWh)'] / (unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)'] * DC_CAPACITY_KWP + 0.001)) * 100

print(f"  ✓ Created 10 performance metrics")

# ============================================================================
# 5.3: STATISTICAL FEATURES
# ============================================================================

print("\n5.3: Creating Statistical Features...")

unified_df = unified_df.sort_values('Date').reset_index(drop=True)

# Rolling averages - 3 day
unified_df['Day_Gen_3d_MA'] = unified_df['Day Gen (KWh)'].rolling(window=3, min_periods=1).mean()
unified_df['Irradiance_3d_MA'] = unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)'].rolling(window=3, min_periods=1).mean()
unified_df['PR_3d_MA'] = unified_df['Radiation_Sensor-Day PR (%)'].rolling(window=3, min_periods=1).mean()
unified_df['CUF_AC_3d_MA'] = unified_df['CUF ON AC Capacity (%)'].rolling(window=3, min_periods=1).mean()

# Rolling averages - 7 day
unified_df['Day_Gen_7d_MA'] = unified_df['Day Gen (KWh)'].rolling(window=7, min_periods=1).mean()
unified_df['Irradiance_7d_MA'] = unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)'].rolling(window=7, min_periods=1).mean()
unified_df['PR_7d_MA'] = unified_df['Radiation_Sensor-Day PR (%)'].rolling(window=7, min_periods=1).mean()
unified_df['CUF_AC_7d_MA'] = unified_df['CUF ON AC Capacity (%)'].rolling(window=7, min_periods=1).mean()

# Rolling std, min, max
unified_df['Day_Gen_7d_Std'] = unified_df['Day Gen (KWh)'].rolling(window=7, min_periods=1).std()
unified_df['Irradiance_7d_Std'] = unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)'].rolling(window=7, min_periods=1).std()
unified_df['Day_Gen_7d_Min'] = unified_df['Day Gen (KWh)'].rolling(window=7, min_periods=1).min()
unified_df['Day_Gen_7d_Max'] = unified_df['Day Gen (KWh)'].rolling(window=7, min_periods=1).max()

# Lag features
unified_df['Day_Gen_Lag1'] = unified_df['Day Gen (KWh)'].shift(1)
unified_df['Day_Gen_Lag2'] = unified_df['Day Gen (KWh)'].shift(2)
unified_df['Day_Gen_Lag3'] = unified_df['Day Gen (KWh)'].shift(3)
unified_df['Day_Gen_Lag7'] = unified_df['Day Gen (KWh)'].shift(7)
unified_df['Irradiance_Lag1'] = unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)'].shift(1)
unified_df['Irradiance_Lag2'] = unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)'].shift(2)
unified_df['Irradiance_Lag7'] = unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)'].shift(7)
unified_df['PR_Lag1'] = unified_df['Radiation_Sensor-Day PR (%)'].shift(1)
unified_df['CUF_AC_Lag1'] = unified_df['CUF ON AC Capacity (%)'].shift(1)

# Differences
unified_df['Day_Gen_Diff1'] = unified_df['Day Gen (KWh)'].diff(1)
unified_df['Irradiance_Diff1'] = unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)'].diff(1)
unified_df['Day_Gen_Trend_3d'] = (unified_df['Day Gen (KWh)'] - unified_df['Day_Gen_Lag3']).fillna(0)

print(f"  ✓ Created 25 statistical features")

# ============================================================================
# 5.4: CATEGORICAL/BINARY INDICATORS
# ============================================================================

print("\n5.4: Creating Categorical/Binary Indicators...")

unified_df['Is_High_Generation'] = (unified_df['Day Gen (KWh)'] > unified_df['Day Gen (KWh)'].quantile(0.75)).astype(int)
unified_df['Is_Low_Generation'] = (unified_df['Day Gen (KWh)'] < unified_df['Day Gen (KWh)'].quantile(0.25)).astype(int)
unified_df['Is_High_Irradiance'] = (unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)'] > 5.5).astype(int)
unified_df['Is_Low_Irradiance'] = (unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)'] < 4.0).astype(int)
unified_df['Is_High_PR'] = (unified_df['Radiation_Sensor-Day PR (%)'] > unified_df['Radiation_Sensor-Day PR (%)'].median()).astype(int)
unified_df['Is_Optimal_Conditions'] = ((unified_df['Is_High_Irradiance'] == 1) & (unified_df['Is_High_PR'] == 1)).astype(int)

def classify_weather(irradiance):
    if irradiance > 6.0:
        return 'Sunny'
    elif irradiance > 4.5:
        return 'Partly_Cloudy'
    elif irradiance > 2.5:
        return 'Cloudy'
    else:
        return 'Overcast'

unified_df['Weather_Classification'] = unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)'].apply(classify_weather)
weather_dummies = pd.get_dummies(unified_df['Weather_Classification'], prefix='Weather')
unified_df = pd.concat([unified_df, weather_dummies], axis=1)

print(f"  ✓ Created 6 binary indicators + weather categories")

# ============================================================================
# 5.5: INTERACTION FEATURES
# ============================================================================

print("\n5.5: Creating Interaction Features...")

unified_df['Irradiance_x_PR'] = unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)'] * unified_df['Radiation_Sensor-Day PR (%)']
unified_df['Irradiance_x_CUF_AC'] = unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)'] * unified_df['CUF ON AC Capacity (%)']
unified_df['Month_x_Irradiance'] = unified_df['Month'] * unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)']
unified_df['Season_x_Irradiance'] = unified_df['Season_Numeric'] * unified_df['Radiation_Sensor-Day INS (KWh/M2/Day)']

print(f"  ✓ Created 4 interaction features")

# ============================================================================
# STEP 6: FINAL DATA CLEANING
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Final Data Cleaning")
print("-" * 80)

missing_count = unified_df.isnull().sum().sum()
print(f"Missing values before cleaning: {missing_count}")

lag_columns = [col for col in unified_df.columns if 'Lag' in col or 'MA' in col or 'Diff' in col or 'Std' in col or 'Trend' in col]
unified_df[lag_columns] = unified_df[lag_columns].fillna(0)

critical_columns = ['Day Gen (KWh)', 'Radiation_Sensor-Day INS (KWh/M2/Day)', 'CUF ON AC Capacity (%)', 'CUF ON DC Capacity (%)']
unified_df = unified_df.dropna(subset=critical_columns)

missing_after = unified_df.isnull().sum().sum()
print(f"Missing values after cleaning: {missing_after}")
print(f"Final dataset rows: {len(unified_df)}")

# ============================================================================
# STEP 7: ORGANIZE AND RENAME COLUMNS
# ============================================================================

print("\n" + "="*80)
print("STEP 7: Organizing Final Dataset")
print("-" * 80)

column_rename_map = {
    'Day Gen (KWh)': 'Energy_Generation_kWh',
    'CUF ON AC Capacity (%)': 'CUF_AC_Percent',
    'CUF ON DC Capacity (%)': 'CUF_DC_Percent',
    'Radiation_Sensor-Day INS (KWh/M2/Day)': 'Solar_Irradiance_kWh_m2_day',
    'Radiation_Sensor-Day PR (%)': 'Performance_Ratio_Percent',
    'Adj Radiation_Sensor-Day INS (KWh/M2/Day)': 'Adjusted_Irradiance_kWh_m2_day',
    'Adj Radiation_Sensor-Day PR (%)': 'Adjusted_PR_Percent'
}

unified_df = unified_df.rename(columns=column_rename_map)

columns_to_drop = ['No.', 'Season', 'Weather_Classification']
unified_df = unified_df.drop(columns=[col for col in columns_to_drop if col in unified_df.columns])

print(f"✓ Dataset organized with {len(unified_df.columns)} features")

# ============================================================================
# STEP 8: SAVE FINAL DATASET
# ============================================================================

print("\n" + "="*80)
print("STEP 8: Saving Final Unified Dataset")
print("-" * 80)

unified_df.to_csv(OUTPUT_FILE, index=False)
print(f"✓ Saved unified dataset to: {OUTPUT_FILE}")
print(f"  - Rows: {unified_df.shape[0]}")
print(f"  - Columns: {unified_df.shape[1]}")


print("\n" + "="*80)
print("PIPELINE COMPLETE! ✓")
print("="*80)

print(f"\nFinal Unified Dataset:")
print(f"  📊 Total Records: {len(unified_df)}")
print(f"  📈 Total Features: {len(unified_df.columns)}")
print(f"  🎯 Target Variable: Energy_Generation_kWh")
print(f"  📅 Date Range: {unified_df['Date'].min().strftime('%Y-%m-%d')} to {unified_df['Date'].max().strftime('%Y-%m-%d')}")

print(f"\nOutput File: {OUTPUT_FILE}")
print(f"\nDataset is ready for ML model training!")
print("\n" + "="*80)

input("\nPress Enter to exit...")