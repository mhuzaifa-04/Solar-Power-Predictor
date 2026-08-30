import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

def train_and_evaluate_models(data_path):
    print("⏳ Loading cleaned dataset...")
    df = pd.read_csv(data_path)

    # 1. Define Features (X) and Target (y)
    feature_cols = [
        'IRRADIANCE', 
        'AMBIENT_TEMPERATURE', 
        'MODULE_TEMPERATURE', 
        'TEMP_DIFF', 
        'HOUR', 
        'MONTH'
    ]
    target_col = 'AC_POWER'

    X = df[feature_cols]
    y = df[target_col]

    # 2. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"📊 Dataset split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")

    # 3. Model 1: Random Forest Regressor
    print("\n🌲 Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    rf_preds = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)

    print(f"   ► Random Forest MAE: {rf_mae:.2f} kW")
    print(f"   ► Random Forest R² Score: {rf_r2:.4f}")

    # 4. Model 2: XGBoost Regressor
    print("\n⚡ Training XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=6, 
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    xgb_preds = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_r2 = r2_score(y_test, xgb_preds)

    print(f"   ► XGBoost MAE: {xgb_mae:.2f} kW")
    print(f"   ► XGBoost R² Score: {xgb_r2:.4f}")

    # 5. Save the Model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/solar_xgboost_model.pkl'

    # Choosing XGBoost as primary production model
    joblib.dump(xgb_model, model_path)
    print(f"\n💾 Model successfully exported to: {model_path}")

if __name__ == "__main__":
    cleaned_file = 'data/cleaned_solar_data.csv'
    if os.path.exists(cleaned_file):
        train_and_evaluate_models(cleaned_file)
    else:
        print("❌ Error: data/cleaned_solar_data.csv not found! Run 1_data_preprocessing.py first.")