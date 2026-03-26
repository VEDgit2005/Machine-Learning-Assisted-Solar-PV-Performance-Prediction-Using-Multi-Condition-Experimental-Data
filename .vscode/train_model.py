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

# ===============================
# 7. FEATURE IMPORTANCE (Random Forest)
# ===============================
import matplotlib.pyplot as plt

rf_model = models["Random Forest"]

importances = rf_model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n📊 Feature Importance:")
print(importance_df)

# Plot Feature Importance
plt.figure()
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.savefig("feature_importance.png")
plt.show()


# ===============================
# 8. PREDICTED VS ACTUAL
# ===============================
y_pred_rf = rf_model.predict(X_test)

plt.figure()
plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual Tilt")
plt.ylabel("Predicted Tilt")
plt.title("Actual vs Predicted (Random Forest)")
plt.savefig("actual_vs_predicted.png")
plt.show()


# ===============================
# 9. ERROR DISTRIBUTION
# ===============================
errors = y_test - y_pred_rf

plt.figure()
plt.hist(errors, bins=50)
plt.title("Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.savefig("error_distribution.png")
plt.show()


# ===============================
# 10. MERGING WITH CLEANED DATASET
# ===============================
try:
    df_real = pd.read_csv("SOLAR_PV_CLEANED_DATASET.csv")
    df_pvlib = pd.read_csv("pvlib_optimal_tilt_dataset.csv")

    # Try datetime merge first
    if 'datetime' in df_real.columns:
        print("\n🔗 Merging using datetime...")

        df_real['datetime'] = pd.to_datetime(df_real['datetime'])
        df_pvlib['datetime'] = pd.to_datetime(df_pvlib['datetime'])

        merged = pd.merge(df_real, df_pvlib, on='datetime', how='inner')

    else:
        print("\n⚠️ No datetime column, merging using month...")

        merged = pd.merge(df_real, df_pvlib, on='month', how='inner')

    merged.to_csv("merged_dataset.csv", index=False)

    print("✅ Merged dataset created: merged_dataset.csv")
    print(merged.head())

except Exception as e:
    print("\n❌ Merge failed:", e)