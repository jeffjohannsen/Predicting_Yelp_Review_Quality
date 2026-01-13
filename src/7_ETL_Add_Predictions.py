"""
Stage 7: Add Model Predictions to Data

This script loads trained models from Stage 5 & 6 and adds their predictions
to the dataset for use in ranking and analysis.

Input:
    - data/model_ready/train.csv, test.csv (from Stage 3)
    - models/final_models/logistic_regression_cv.joblib (from Stage 5)
    - models/final_models/linear_regression.joblib (from Stage 6)

Output:
    - data/final_predict/train_with_predictions.csv
    - data/final_predict/test_with_predictions.csv

Usage:
    python src/7_ETL_Add_Predictions.py
"""

import logging
import time
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler

from config import PathConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pandas display options
pd.set_option("display.float_format", lambda x: "%.5f" % x)
pd.set_option("display.max_columns", 200)

# Column configuration
EXCLUDE_COLS = [
    "review_id",
    "T1_REG_review_total_ufc",
    "T2_CLS_ufc_>0",
    "T3_CLS_ufc_level",
    "T4_REG_ufc_TD",
    "T5_CLS_ufc_level_TD",
    "T6_REG_ufc_TDBD",
]

# Feature group used for predictions (must match what models were trained on)
FEATURE_COLS = [
    "NB_prob", "svm_pred", "ft_prob",
    "lda_t1", "lda_t2", "lda_t3", "lda_t4", "lda_t5",
]


def load_data(train_nrows: int = None, test_nrows: int = None):
    """
    Load train and test data from model_ready directory.

    Args:
        train_nrows: Number of training rows to load (None = all)
        test_nrows: Number of test rows to load (None = all)

    Returns:
        tuple: (train_df, test_df)
    """
    start = time.perf_counter()
    model_ready_dir = PathConfig.get_model_ready_dir()

    train = pd.read_csv(model_ready_dir / "train.csv", nrows=train_nrows)
    test = pd.read_csv(model_ready_dir / "test.csv", nrows=test_nrows)

    elapsed = time.perf_counter() - start
    logger.info(f"Data loaded in {elapsed:.2f} seconds")
    logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")

    return train, test


def prepare_features(train, test):
    """
    Prepare and scale features for prediction.

    Args:
        train: Training DataFrame
        test: Test DataFrame

    Returns:
        tuple: (X_train_scaled, X_test_scaled)
    """
    # Filter to only columns that exist in the data
    available_cols = [c for c in FEATURE_COLS if c in train.columns]
    if len(available_cols) < len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(available_cols)
        logger.warning(f"Missing columns: {missing}")

    X_train = train[available_cols]
    X_test = test[available_cols]

    logger.info(f"Using {len(available_cols)} features for prediction")

    # Scale features (models were trained on scaled data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def add_predictions(train, test, X_train, X_test):
    """
    Load models and add predictions to DataFrames.

    Args:
        train: Training DataFrame
        test: Test DataFrame
        X_train: Scaled training features
        X_test: Scaled test features

    Returns:
        tuple: (train_with_preds, test_with_preds)
    """
    models_dir = PathConfig.get_final_models_dir()

    # Classification predictions
    clf_model_path = models_dir / "logistic_regression_cv.joblib"
    if clf_model_path.exists():
        logger.info(f"Loading classification model from {clf_model_path}")
        clf_model = load(clf_model_path)

        clf_pred_train = clf_model.predict_proba(X_train)[:, 1]
        clf_pred_test = clf_model.predict_proba(X_test)[:, 1]

        train = train.copy()
        test = test.copy()
        train.insert(1, "clf_pred_proba", clf_pred_train)
        test.insert(1, "clf_pred_proba", clf_pred_test)

        logger.info("Classification predictions added")
    else:
        logger.warning(f"Classification model not found at {clf_model_path}")

    # Regression predictions
    reg_model_path = models_dir / "linear_regression.joblib"
    if reg_model_path.exists():
        logger.info(f"Loading regression model from {reg_model_path}")
        reg_model = load(reg_model_path)

        reg_pred_train = reg_model.predict(X_train)
        reg_pred_test = reg_model.predict(X_test)

        train.insert(2, "reg_pred", reg_pred_train)
        test.insert(2, "reg_pred", reg_pred_test)

        logger.info("Regression predictions added")
    else:
        logger.warning(f"Regression model not found at {reg_model_path}")

    return train, test


def main():
    """Main pipeline."""
    logger.info("=" * 60)
    logger.info("Stage 7: Add Model Predictions")
    logger.info("=" * 60)

    # Load data
    train, test = load_data()

    # Prepare features
    X_train, X_test = prepare_features(train, test)

    # Add predictions
    train_pred, test_pred = add_predictions(train, test, X_train, X_test)

    # Save to final_predict directory
    output_dir = PathConfig.get_final_predict_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_with_predictions.csv"
    test_path = output_dir / "test_with_predictions.csv"

    train_pred.to_csv(train_path, index=False)
    test_pred.to_csv(test_path, index=False)

    logger.info(f"Train with predictions saved to {train_path}")
    logger.info(f"Test with predictions saved to {test_path}")

    # Show sample of predictions
    logger.info("\nSample predictions (test set):")
    pred_cols = ["review_id", "clf_pred_proba", "reg_pred", "T2_CLS_ufc_>0", "T4_REG_ufc_TD"]
    available_pred_cols = [c for c in pred_cols if c in test_pred.columns]
    print(test_pred[available_pred_cols].head(10).to_string())

    logger.info("=" * 60)
    logger.info("Stage 7 Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
