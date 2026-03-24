import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

# ---------------------------
# 1. LOAD DATA
# ---------------------------
df = pd.read_csv("pvlib_optimal_tilt_dataset.csv")

# ---------------------------
# 2. FEATURES & TARGET
# ---------------------------
features = [
    'zenith', 'azimuth',
    'ghi', 'dni', 'dhi',
    'airmass', 'month', 'hour'
]

target = 'optimal_tilt'

X = df[features]
y = df[target]

# Handle NaN (airmass night values already removed, but safe)
X = X.fillna(0)

# ---------------------------
# 3. TRAIN TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 4. MODELS
# ---------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
}

# ---------------------------
# 5. TRAIN & EVALUATE
# ---------------------------
results = []

for name, model in models.items():
    print(f"\n🔹 Training {name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"{name} Results:")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")

# ---------------------------
# 6. RESULTS TABLE
# ---------------------------
results_df = pd.DataFrame(results)
print("\n📊 Final Comparison:")
print(results_df)

results_df.to_csv("model_comparison_results.csv", index=False)