import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def clean_solar_pv_dataset(input_file, output_file):
    """
    Comprehensive data cleaning for Solar PV dataset
    """
    print("=" * 80)
    print("SOLAR PV DATASET CLEANING PIPELINE")
    print("=" * 80)
    
    # Load dataset
    df = pd.read_csv(input_file)
    print(f"\n✓ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    initial_rows = len(df)
    
    # ============================================================================
    # STEP 1: Remove duplicate columns
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Removing Duplicate Columns")
    print("=" * 80)
    
    # Identify duplicate columns
    duplicate_cols = []
    if 'no._y' in df.columns:
        duplicate_cols.append('no._y')
    if 'day_gen__kwh' in df.columns and 'day_gen_kwh' in df.columns:
        duplicate_cols.append('day_gen__kwh')
    if 'radiation_sensor-day_ins_kwh/m2/day_y' in df.columns:
        duplicate_cols.append('radiation_sensor-day_ins_kwh/m2/day_y')
    if 'radiation_sensor-day_pr_%_y' in df.columns:
        duplicate_cols.append('radiation_sensor-day_pr_%_y')
    
    df = df.drop(columns=duplicate_cols, errors='ignore')
    print(f"✓ Removed {len(duplicate_cols)} duplicate columns: {duplicate_cols}")
    
    # Rename remaining columns for clarity
    rename_map = {
        'no._x': 'id',
        'radiation_sensor-day_ins_kwh/m2/day_x': 'radiation_kwh_m2_day',
        'radiation_sensor-day_pr_%_x': 'performance_ratio_pct',
        'adj_radiation_sensor-day_ins_kwh/m2/day': 'adj_radiation_kwh_m2_day',
        'adj_radiation_sensor-day_pr_%': 'adj_performance_ratio_pct'
    }
    df = df.rename(columns=rename_map)
    print(f"✓ Renamed columns for better clarity")
    
    # ============================================================================
    # STEP 2: Convert data types
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Converting Data Types")
    print("=" * 80)
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print(f"✓ Converted 'date' to datetime")
    
    # Convert percentage columns from object to float
    pct_cols = [col for col in df.columns if 'pct' in col]
    for col in pct_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"✓ Converted {len(pct_cols)} percentage columns to numeric")
    
    # ============================================================================
    # STEP 3: Handle missing values
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Handling Missing Values")
    print("=" * 80)
    
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before: {missing_before}")
    
    # Remove rows with missing dates (critical field)
    rows_before = len(df)
    df = df.dropna(subset=['date'])
    print(f"✓ Removed {rows_before - len(df)} rows with missing dates")
    
    # For numeric columns, use interpolation or forward fill
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['id', 'hour', 'day', 'month', 'year']:
            # Try interpolation first
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            # Fill any remaining NaNs with forward fill, then backward fill
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    missing_after = df.isnull().sum().sum()
    print(f"✓ Missing values after cleaning: {missing_after}")
    print(f"✓ Filled/interpolated missing values in numeric columns")
    
    # ============================================================================
    # STEP 4: Remove outliers
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Handling Outliers")
    print("=" * 80)
    
    # Identify and cap extreme outliers using IQR method
    outlier_cols = ['day_gen_kwh', 'cuf_on_ac_capacity_%', 'cuf_on_dc_capacity_%']
    outliers_removed = 0
    
    for col in outlier_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            # Cap outliers instead of removing rows
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            outliers_removed += outliers
            
            print(f"✓ {col}: Capped {outliers} outliers to [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    print(f"✓ Total outliers capped: {outliers_removed}")
    
    # ============================================================================
    # STEP 5: Fix inconsistencies
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Fixing Inconsistencies")
    print("=" * 80)
    
    # Ensure all negative values are set to 0 (solar generation can't be negative)
    for col in ['day_gen_kwh', 'radiation_kwh_m2_day', 'adj_radiation_kwh_m2_day']:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                df[col] = df[col].clip(lower=0)
                print(f"✓ {col}: Corrected {neg_count} negative values to 0")
    
    # Ensure performance ratios are within valid range (0-100%)
    for col in pct_cols:
        if col in df.columns:
            invalid = ((df[col] < 0) | (df[col] > 100)).sum()
            if invalid > 0:
                df[col] = df[col].clip(lower=0, upper=100)
                print(f"✓ {col}: Corrected {invalid} invalid percentage values")
    
    # Fix hour column (all zeros - should be removed or properly populated)
    if 'hour' in df.columns:
        if df['hour'].nunique() == 1 and df['hour'].iloc[0] == 0:
            df = df.drop(columns=['hour'])
            print("✓ Removed 'hour' column (all zeros - not useful)")
    
    # ============================================================================
    # STEP 6: Add derived features
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Adding Derived Features")
    print("=" * 80)
    
    # Add day of week
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_name'] = df['date'].dt.day_name()
    
    # Add quarter
    df['quarter'] = df['date'].dt.quarter
    
    # Add is_weekend flag
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Add season (for Northern Hemisphere)
    df['season'] = df['month'].apply(lambda x: 
        'Winter' if x in [12, 1, 2] else
        'Spring' if x in [3, 4, 5] else
        'Summer' if x in [6, 7, 8] else
        'Autumn'
    )
    
    print("✓ Added: day_of_week, day_name, quarter, is_weekend, season")
    
    # ============================================================================
    # STEP 7: Sort and reorder columns
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 7: Final Organization")
    print("=" * 80)
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Reorder columns logically
    id_cols = ['id', 'date', 'year', 'month', 'day', 'day_of_week', 'day_name', 'quarter', 'season', 'is_weekend']
    energy_cols = ['day_gen_kwh']
    radiation_cols = [col for col in df.columns if 'radiation' in col]
    performance_cols = [col for col in df.columns if 'performance' in col or 'cuf' in col]
    remaining_cols = [col for col in df.columns if col not in id_cols + energy_cols + radiation_cols + performance_cols]
    
    column_order = id_cols + energy_cols + radiation_cols + performance_cols + remaining_cols
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    print("✓ Sorted data by date")
    print("✓ Reordered columns logically")
    
    # ============================================================================
    # STEP 8: Quality checks and summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 8: Final Quality Checks")
    print("=" * 80)
    
    final_rows = len(df)
    print(f"✓ Final dataset: {final_rows} rows, {len(df.columns)} columns")
    print(f"✓ Rows retained: {final_rows}/{initial_rows} ({final_rows/initial_rows*100:.1f}%)")
    print(f"✓ Missing values: {df.isnull().sum().sum()}")
    print(f"✓ Duplicate rows: {df.duplicated().sum()}")
    
    # Remove any duplicate rows
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
        print(f"✓ Removed {df.duplicated().sum()} duplicate rows")
    
    # ============================================================================
    # STEP 9: Save cleaned dataset
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 9: Saving Cleaned Dataset")
    print("=" * 80)
    
    df.to_csv(output_file, index=False)
    print(f"✓ Saved cleaned dataset to: {output_file}")
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total days: {(df['date'].max() - df['date'].min()).days + 1}")
    
    print("\n" + "=" * 80)
    print("✓ CLEANING COMPLETE!")
    print("=" * 80)
    
    return df

# Run the cleaning pipeline
if __name__ == "__main__":
    input_file = '/mnt/user-data/uploads/1770292412784_SOLAR_PV_ML_READY_DATASET.csv'
    output_file = '/mnt/user-data/outputs/SOLAR_PV_CLEANED_DATASET.csv'
    
    cleaned_df = clean_solar_pv_dataset(input_file, output_file)
    
    # Display sample of cleaned data
    print("\n" + "=" * 80)
    print("SAMPLE OF CLEANED DATA (First 10 rows)")
    print("=" * 80)
    print(cleaned_df.head(10))
    
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)
    print(cleaned_df.describe())
