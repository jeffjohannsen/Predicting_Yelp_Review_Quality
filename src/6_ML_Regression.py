"""
Stage 6: Regression Models

This script trains regression models for predicting review vote counts
(time-discounted useful/funny/cool votes).

Input: data/model_ready/train.csv, test.csv (from Stage 3)
Output: models/final_models/linear_regression.joblib

Usage:
    python src/6_ML_Regression.py
"""

import logging
import time
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler

from config import PathConfig

# Optional MLflow tracking
try:
    import mlflow
    mlflow.sklearn.autolog()
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pandas display options
pd.set_option("display.float_format", lambda x: "%.5f" % x)
pd.set_option("display.max_columns", 200)

# Column configuration - Regression target
TARGET_COL = "T4_REG_ufc_TD"  # Time-discounted vote count
EXCLUDE_COLS = [
    "review_id",
    "T1_REG_review_total_ufc",
    "T2_CLS_ufc_>0",
    "T3_CLS_ufc_level",
    "T4_REG_ufc_TD",
    "T5_CLS_ufc_level_TD",
    "T6_REG_ufc_TDBD",
]

# Feature groups for experimentation
FEATURE_GROUPS = {
    "submodels": [
        "NB_prob", "svm_pred", "ft_prob",
        "lda_t1", "lda_t2", "lda_t3", "lda_t4", "lda_t5",
    ],
    "other": [
        "review_stars", "grade_level", "polarity", "subjectivity",
    ],
    "basic_text": [
        "word_cnt", "character_cnt", "num_cnt", "uppercase_cnt",
        "sentence_cnt", "lexicon_cnt", "syllable_cnt", "avg_word_len",
        "token_cnt", "stopword_cnt", "stopword_pct", "ent_cnt", "ent_pct",
    ],
    "top_features": [
        "svm_pred", "ft_prob", "NB_prob", "token_cnt", "review_stars",
        "polarity", "subjectivity", "grade_level", "character_cnt",
        "avg_word_len", "lda_t1", "lda_t2", "lda_t3", "lda_t4", "lda_t5",
    ],
}


class RegressionModelTrainer:
    """Wrapper class for training and evaluating regression models."""

    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.predictions = None

    def fit(self, X_train, y_train):
        """Fit the model with optional MLflow tracking."""
        start = time.perf_counter()
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=self.model_name):
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        logger.info(f"{self.model_name} training completed in {elapsed/60:.2f} minutes")

    def predict(self, X_test):
        """Generate predictions."""
        start = time.perf_counter()
        self.predictions = self.model.predict(X_test)
        elapsed = time.perf_counter() - start
        logger.info(f"{self.model_name} predictions completed in {elapsed:.2f} seconds")

    def evaluate(self, y_test):
        """Evaluate model performance with regression metrics."""
        r2 = r2_score(y_test, self.predictions)
        mae = mean_absolute_error(y_test, self.predictions)
        explained_var = explained_variance_score(y_test, self.predictions)

        logger.info(f"{self.model_name} R² Score: {r2:.4f}")
        logger.info(f"{self.model_name} MAE: {mae:.4f}")
        logger.info(f"{self.model_name} Explained Variance: {explained_var:.4f}")

        return {"r2": r2, "mae": mae, "explained_variance": explained_var}

    def save(self, output_path: Path):
        """Save model to disk."""
        dump(self.model, output_path)
        logger.info(f"{self.model_name} saved to {output_path}")


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


def prepare_features(train, test, feature_group: str = "all", scale: bool = True):
    """
    Prepare features for training.

    Args:
        train: Training DataFrame
        test: Test DataFrame
        feature_group: Which feature group to use ('all', 'submodels', 'top_features', etc.)
        scale: Whether to apply StandardScaler

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Determine which features to use
    if feature_group == "all":
        feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]
    elif feature_group in FEATURE_GROUPS:
        feature_cols = FEATURE_GROUPS[feature_group]
    else:
        raise ValueError(f"Unknown feature group: {feature_group}")

    # Filter to only columns that exist in the data
    available_cols = [c for c in feature_cols if c in train.columns]
    if len(available_cols) < len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        logger.warning(f"Missing columns: {missing}")

    X_train = train[available_cols]
    X_test = test[available_cols]
    y_train = train[TARGET_COL]
    y_test = test[TARGET_COL]

    logger.info(f"Using {len(available_cols)} features from group '{feature_group}'")
    logger.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    logger.info(f"Target: {TARGET_COL} (regression)")

    # Optional scaling
    if scale:
        start = time.perf_counter()
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=available_cols,
            index=X_train.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=available_cols,
            index=X_test.index
        )
        elapsed = time.perf_counter() - start
        logger.info(f"Data scaled in {elapsed:.2f} seconds")

    return X_train, X_test, y_train, y_test


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Stage 6: Regression Models")
    logger.info("=" * 60)

    # Load data
    train, test = load_data()

    # Prepare features (using submodels - the most predictive features)
    X_train, X_test, y_train, y_test = prepare_features(
        train, test,
        feature_group="submodels",
        scale=True
    )

    # Train Linear Regression
    logger.info("Training Linear Regression...")
    lin_reg = LinearRegression(n_jobs=-1)

    trainer = RegressionModelTrainer(lin_reg, "LinearRegression")
    trainer.fit(X_train, y_train)
    trainer.predict(X_test)
    results = trainer.evaluate(y_test)

    # Save model
    output_dir = PathConfig.get_final_models_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(output_dir / "linear_regression.joblib")

    logger.info("=" * 60)
    logger.info(f"Stage 6 Complete - Final R²: {results['r2']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
