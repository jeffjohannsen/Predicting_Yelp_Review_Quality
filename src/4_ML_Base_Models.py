"""
Stage 4.1: Base Model Training

Trains baseline classifiers for review quality prediction:
- Logistic Regression (with StandardScaler)
- Decision Tree
- Random Forest
- XGBoost

These base models establish baseline performance before hyperparameter tuning.
Models are saved for later comparison and ensemble building.

Input: Stage 3 output (model-ready CSV files)
Output: Trained models in models/base_models/

Pipeline: Stage 1 → Stage 2 → Stage 3 → Stage 4 (this) → Stage 5+
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import PathConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Target column for binary classification
TARGET_COL = "T2_CLS_ufc_>0"

# Columns to exclude from features
EXCLUDE_COLS = [
    "review_id",
    "T1_REG_review_total_ufc",
    "T2_CLS_ufc_>0",
    "T3_CLS_ufc_level",
    "T4_REG_ufc_TD",
    "T5_CLS_ufc_level_TD",
    "T6_REG_ufc_TDBD",
]


def load_data(nrows: int = None) -> tuple:
    """Load train and test data from Stage 3 output.

    Args:
        nrows: Number of rows to load (None for all)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    model_ready_dir = PathConfig.get_model_ready_dir()

    logger.info(f"Loading data from {model_ready_dir}")

    train = pd.read_csv(model_ready_dir / "train.csv", nrows=nrows)
    test = pd.read_csv(model_ready_dir / "test.csv", nrows=nrows)

    logger.info(f"Train shape: {train.shape}")
    logger.info(f"Test shape: {test.shape}")

    # Separate features and target
    feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]

    X_train = train[feature_cols]
    X_test = test[feature_cols]
    y_train = train[TARGET_COL]
    y_test = test[TARGET_COL]

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Target distribution (train): {y_train.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


class BaseModelTrainer:
    """Wrapper for training and evaluating base models."""

    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.predictions = None
        self.predict_proba = None
        self.train_time = 0
        self.metrics = {}

    def fit(self, X_train, y_train):
        """Train the model."""
        logger.info(f"Training {self.model_name}...")
        start = time.perf_counter()
        self.model.fit(X_train, y_train)
        self.train_time = time.perf_counter() - start
        logger.info(f"  Training completed in {self.train_time:.2f} seconds")

    def predict(self, X_test):
        """Generate predictions."""
        self.predictions = self.model.predict(X_test)
        self.predict_proba = self.model.predict_proba(X_test)[:, 1]

    def evaluate(self, y_test):
        """Evaluate model performance."""
        accuracy = accuracy_score(y_test, self.predictions)
        auc = roc_auc_score(y_test, self.predict_proba)

        self.metrics = {
            "accuracy": accuracy,
            "auc": auc,
            "train_time": self.train_time,
        }

        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  AUC: {auc:.4f}")

        # Detailed classification report
        report = classification_report(y_test, self.predictions)
        logger.info(f"  Classification Report:\n{report}")

        return self.metrics

    def save(self, output_dir: Path):
        """Save model to disk."""
        model_path = output_dir / f"{self.model_name.lower().replace(' ', '_')}.joblib"
        dump(self.model, model_path)
        logger.info(f"  Saved to {model_path}")


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Stage 4.1: Base Model Training")
    logger.info("=" * 60)

    # Setup paths
    output_dir = PathConfig.get_base_models_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Scale features for Logistic Regression
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    scaler_path = output_dir / "standard_scaler.joblib"
    dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")

    # Define models
    models = [
        (LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
         "Logistic Regression", True),  # True = use scaled data
        (DecisionTreeClassifier(random_state=42),
         "Decision Tree", False),
        (RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100),
         "Random Forest", False),
        (XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
         "XGBoost", False),
    ]

    # Train and evaluate each model
    results = []
    for model, name, use_scaled in models:
        logger.info("-" * 40)
        logger.info(f">>> {name}")

        trainer = BaseModelTrainer(model, name)

        # Use scaled or unscaled data
        if use_scaled:
            trainer.fit(X_train_scaled, y_train)
            trainer.predict(X_test_scaled)
        else:
            trainer.fit(X_train, y_train)
            trainer.predict(X_test)

        metrics = trainer.evaluate(y_test)
        trainer.save(output_dir)

        results.append({
            "model": name,
            **metrics
        })

    # Summary
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("auc", ascending=False)
    logger.info(f"\n{results_df.to_string(index=False)}")

    # Save results
    results_path = output_dir / "base_model_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to {results_path}")

    logger.info("=" * 60)
    logger.info("Stage 4.1 Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
