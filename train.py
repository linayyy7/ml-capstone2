"""
Production training script for Remote Worker Productivity Prediction

This script:
- Loads the raw dataset (not cleaned, not encoded)
- Performs data cleaning and feature selection
- Applies one-hot encoding
- Trains multiple regression models
- Selects the best model based on validation RMSE
- Evaluates once on the test set
- Saves the final model to model.bin
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

import seaborn as sns
import matplotlib.pyplot as plt



# Configuration

DATA_PATH = "data/remote_worker_productivity_1000.csv"
MODEL_PATH = "model.bin"
TARGET = "productivity_score"
RANDOM_STATE = 42


def rmse(y_true, y_pred):
    """Compute Root Mean Squared Error"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def clean_data(df):
    """
    Clean raw dataset and apply one-hot encoding.

    Steps:
    - Drop identifier and leakage columns
    - Select relevant features
    - Handle missing values
    - Apply one-hot encoding to categorical features
    """

    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Drop identifier and leakage columns
    drop_cols = ["worker_id", "productivity_label"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Categorical features
    categorical_col= ["location_type", "industry_sector"]

    # Numerical features
    numerical_col = [
        "age",
        "experience_years",
        "average_daily_work_hours",
        "break_frequency_per_day",
        "late_task_ratio",
        "calendar_scheduled_usage",
        "focus_time_minutes",
        "tool_usage_frequency",
        "automated_task_count",
        "ai_assisted_planning",
        "real_time_feedback_score",
    ]

    # Keep only modeling columns
    df = df[categorical_col + numerical_col + [TARGET]].copy()

    # Handle missing values
    df = df.dropna()

    return df, categorical_col, numerical_col



# Main training pipeline


def main():
    print("📥 Loading raw dataset...")
    df_raw = pd.read_csv(DATA_PATH)

    print("🧹 Cleaning data...")
    df, categorical_col, numerical_col = clean_data(df_raw)

    print("Final dataset shape:", df.shape)

   
    # Split the data into training, validation, and test sets
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=RANDOM_STATE)
    # Print the sizes of each dataset
    len(df_train), len(df_val), len(df_test)

    # Reset the index for all split datasets to avoid index conflicts
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_full_train = df_full_train.reset_index(drop=True)



    print("Train size:", df_train.shape)
    print("Validation size:", df_val.shape)
    print("Test size:", df_test.shape)

    # Extract target variable (y) from each dataset
    y_train = df_train.productivity_score.values
    y_val = df_val.productivity_score.values
    y_test = df_test.productivity_score.values

    # Remove the target column from feature sets
    del df_train["productivity_score"]
    del df_val["productivity_score"]
    del df_test["productivity_score"]


    # One-Hot Encoding
    print("🔢 Applying DictVectorizer...")
    dv = DictVectorizer(sparse=False)

    train_dict = df_train[categorical_col + numerical_col].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    val_dict = df_val[categorical_col + numerical_col].to_dict(orient='records')
    X_val = dv.transform(val_dict)
    test_dict = df_test[categorical_col + numerical_col].to_dict(orient='records')
    X_test = dv.transform(test_dict)

    print("Encoded train shape:", X_train.shape)
    print("Encoded val shape:", X_val.shape)
    print("Encoded test shape:", X_test.shape)


    # Models

    # Linear Regression
    print("\n🔹 Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    rmse_lr = rmse(y_val, lr.predict(X_val))

    print("🔹 Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    rmse_ridge = rmse(y_val, ridge.predict(X_val))

    print("🔹 Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rmse_rf = rmse(y_val, rf.predict(X_val))


    # Model Comparison
    print("\n📊 Validation RMSE:")
    print(f"Linear Regression : {rmse_lr:.4f}")
    print(f"Ridge Regression  : {rmse_ridge:.4f}")
    print(f"Random Forest    : {rmse_rf:.4f}")

    #Final Evaluation on Test Set

    # Select best model
    best_model = rf
    print("\n✅ Selected model: Random Forest")


    # Final test evaluation
    print("\n🧪 Evaluating on test set...")
    test_rmse = rmse(y_test, best_model.predict(X_test))
    print(f"Test RMSE: {test_rmse:.4f}")


    # Save model
    print("\n💾 Saving model and DictVectorizer...")
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": best_model,
                "dv": dv
            },
            f
        )

    print(f"Saved model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
