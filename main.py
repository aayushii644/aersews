# ==========================================
# Concrete Strength Prediction - Full ML Pipeline
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

# ==========================================
# 1. Load Dataset
# ==========================================

import os

DATA_PATH = "Concrete_Data.xls"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at '{DATA_PATH}'. Please put the file in the project root.")

try:
    df = pd.read_excel(DATA_PATH)
except ImportError:
    raise ImportError(
        "Reading Excel files requires an optional dependency (xlrd for .xls or openpyxl for .xlsx)."
        " Install with: pip install xlrd openpyxl"
    )
except Exception:
    # As a last resort, attempt to read as CSV with a permissive encoding and show a helpful error if that fails.
    try:
        df = pd.read_csv(DATA_PATH, encoding='latin1')
    except Exception as exc:
        raise RuntimeError(
            "Failed to read dataset as Excel or CSV. Ensure the file is a valid Excel (.xls/.xlsx) or CSV file."
        ) from exc

print("Dataset Shape:", df.shape)
print(df.head())

# Quick verification mode: set environment variable QUICK_RUN=0 to run full search.
# Default is QUICK_RUN=1 for quicker verification runs.
QUICK_RUN = os.environ.get("QUICK_RUN", "1") == "1"

# ==========================================
# 2. Define Features and Target
# ==========================================

# Normalize column names (trim whitespace)
df.columns = [c.strip() for c in df.columns]

# Try to locate the target column by common name; fallback to last column
target_col = None
for col in df.columns:
    if col.lower().startswith("concrete compressive strength"):
        target_col = col
        break

if target_col is None:
    target_col = df.columns[-1]
    print(f"Warning: couldn't find explicit target column name; using '{target_col}' as target.")

X = df.drop(target_col, axis=1)
y = df[target_col]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 3. Linear Regression (Pipeline)
# ==========================================

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

# ==========================================
# 4. Random Forest with Hyperparameter Tuning
# ==========================================

rf = RandomForestRegressor(random_state=42)

if QUICK_RUN:
    param_grid_rf = {
        'n_estimators': [100],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    rf_n_jobs = 1
else:
    param_grid_rf = {
        'n_estimators': [200, 400],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_n_jobs = -1

grid_rf = GridSearchCV(
    rf,
    param_grid_rf,
    cv=5,
    scoring='r2',
    n_jobs=rf_n_jobs
)

grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# ==========================================
# 5. XGBoost with Hyperparameter Tuning
# ==========================================

xgb = XGBRegressor(
    objective='reg:squarederror',
    random_state=42
)

param_grid_xgb = {
    'n_estimators': [300, 500],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

if QUICK_RUN:
    param_grid_xgb = {
        'n_estimators': [100],
        'learning_rate': [0.1],
        'max_depth': [3],
        'subsample': [1.0],
        'colsample_bytree': [1.0]
    }
    xgb_n_jobs = 1
else:
    param_grid_xgb = {
        'n_estimators': [300, 500],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb_n_jobs = -1

grid_xgb = GridSearchCV(
    xgb,
    param_grid_xgb,
    cv=5,
    scoring='r2',
    n_jobs=xgb_n_jobs
)

grid_xgb.fit(X_train, y_train)

best_xgb = grid_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)

# ==========================================
# 6. Model Evaluation
# ==========================================

def evaluate_model(name, y_true, y_pred):
    print(f"\n{name}")
    print("-" * 30)
    print("MAE :", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R2  :", r2_score(y_true, y_pred))


evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("XGBoost", y_test, y_pred_xgb)

print("\nBest RF Parameters:", grid_rf.best_params_)
print("Best XGB Parameters:", grid_xgb.best_params_)

# ==========================================
# 7. Feature Importance (XGBoost)
# ==========================================

importances = best_xgb.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances)
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
# Save plot non-interactively to outputs/
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "feature_importance.png")
plt.savefig(out_path)
plt.close()
print(f"Saved feature importance plot to {out_path}")

# ==========================================
# 8. Predict New Concrete Sample
# ==========================================

new_sample = pd.DataFrame(
    [[540, 0, 0, 162, 2.5, 1040, 676, 28]],
    columns=X.columns
)

prediction = best_xgb.predict(new_sample)

print("\nPredicted Concrete Strength (MPa):", prediction[0])

# ==========================================
# END
# ==========================================