import pandas as pd

# =========================
# 1. LOAD DATASETS
# =========================

weather_path = "SRM_ANNEXURE_CAMPUS_837.1___Weather ___(Day Reports 30_08_2024 To 20_01_2026)_.xlsx"
solar_path = "SRM_ANNEXURE_CAMPUS_837.1_Solar_Meter(Day Reports_30_08_2024 To 20_01_2026)_.xlsx"

weather_df = pd.read_excel(weather_path)
solar_df = pd.read_excel(solar_path)

# =========================
# 2. STANDARDIZE COLUMNS
# =========================

weather_df.columns = (
    weather_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
)

solar_df.columns = (
    solar_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
)

# =========================
# 3. DATE/TIME HANDLING
# =========================

weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce", dayfirst=True)
solar_df["date"] = pd.to_datetime(solar_df["date"], errors="coerce", dayfirst=True)

# Remove rows with invalid dates
weather_df = weather_df.dropna(subset=["date"])
solar_df = solar_df.dropna(subset=["date"])

# =========================
# 4. REMOVE DUPLICATES
# =========================

weather_df = weather_df.drop_duplicates()
solar_df = solar_df.drop_duplicates()

# =========================
# 5. HANDLE MISSING VALUES
# =========================
# Numeric columns → Interpolation
# Categorical columns → Forward fill

weather_df = weather_df.sort_values("date")
solar_df = solar_df.sort_values("date")

weather_df = weather_df.interpolate(method="linear").ffill().bfill()
solar_df = solar_df.interpolate(method="linear").ffill().bfill()

# =========================
# 6. MERGE DATASETS
# =========================

merged_df = pd.merge(
    weather_df,
    solar_df,
    on="date",
    how="inner"
)

# =========================
# 7. FEATURE ENGINEERING
# =========================

merged_df["hour"] = merged_df["date"].dt.hour
merged_df["day"] = merged_df["date"].dt.day
merged_df["month"] = merged_df["date"].dt.month
merged_df["year"] = merged_df["date"].dt.year

# =========================
# 8. FINAL CLEAN CHECK
# =========================

merged_df = merged_df.dropna()
merged_df.reset_index(drop=True, inplace=True)

# =========================
# 9. SAVE ML-READY DATASET
# =========================

output_file = "SOLAR_PV_ML_READY_DATASET.csv"
merged_df.to_csv(output_file, index=False)

print("Dataset successfully created!")
print("Shape:", merged_df.shape)
print("Saved as:", output_file)
