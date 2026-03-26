import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv("merged_dataset.csv")

# ===============================
# 2. HANDLE DATETIME (IMPORTANT)
# ===============================
if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])

    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour

    df = df.drop(columns=['datetime'])

# ===============================
# 3. CLEAN DATA
# ===============================
df = df.fillna(0)

# Keep only numeric columns
df = df.select_dtypes(include=[np.number])

# ===============================
# 4. DEFINE TARGET
# ===============================
target = 'max_power'

# ===============================
# 5. FEATURE SELECTION
# ===============================
exclude_cols = ['max_power', 'optimal_tilt']

# Model A → Real features only
real_features = [
    col for col in df.columns
    if col not in exclude_cols
    and col not in ['zenith', 'azimuth', 'ghi', 'dni', 'dhi', 'airmass']
]

# Model B → Real + PVLib features
pvlib_features = ['zenith', 'azimuth', 'ghi', 'dni', 'dhi', 'airmass']

enhanced_features = real_features + pvlib_features

# ===============================
# 6. PREPARE DATA
# ===============================
X_A = df[real_features]
X_B = df[enhanced_features]
y = df[target]

# Train-test split
X_train_A, X_test_A, y_train, y_test = train_test_split(
    X_A, y, test_size=0.2, random_state=42
)

X_train_B, X_test_B, _, _ = train_test_split(
    X_B, y, test_size=0.2, random_state=42
)

# ===============================
# 7. MODELS
# ===============================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
}

# ===============================
# 8. TRAIN & COMPARE
# ===============================
results = []

for name, model in models.items():
    print(f"\n🔹 Training {name}...")

    # ---- Model A ----
    model.fit(X_train_A, y_train)
    pred_A = model.predict(X_test_A)

    # ---- Model B ----
    model.fit(X_train_B, y_train)
    pred_B = model.predict(X_test_B)

    # Metrics A
    mae_A = mean_absolute_error(y_test, pred_A)
    rmse_A = np.sqrt(mean_squared_error(y_test, pred_A))
    r2_A = r2_score(y_test, pred_A)

    # Metrics B
    mae_B = mean_absolute_error(y_test, pred_B)
    rmse_B = np.sqrt(mean_squared_error(y_test, pred_B))
    r2_B = r2_score(y_test, pred_B)

    results.append({
        "Model": name,
        "MAE_A": mae_A,
        "RMSE_A": rmse_A,
        "R2_A": r2_A,
        "MAE_B": mae_B,
        "RMSE_B": rmse_B,
        "R2_B": r2_B
    })

    print(f"\n{name} Results:")
    print(f"Model A (Real Only)  -> RMSE: {rmse_A:.3f}, R2: {r2_A:.3f}")
    print(f"Model B (Enhanced)   -> RMSE: {rmse_B:.3f}, R2: {r2_B:.3f}")

# ===============================
# 9. FINAL RESULTS
# ===============================
results_df = pd.DataFrame(results)

print("\n📊 FINAL COMPARISON:")
print(results_df)

results_df.to_csv("final_model_comparison.csv", index=False)

print("\n✅ Results saved as final_model_comparison.csv")