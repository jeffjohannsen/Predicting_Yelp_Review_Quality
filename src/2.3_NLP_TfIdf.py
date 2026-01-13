"""
Stage 2.3: TF-IDF Vectorization and ML Model Predictions

Trains TF-IDF vectorizer and classification models to generate prediction features:
- NB_prob: Naive Bayes probability of positive class
- svm_pred: SVM raw prediction score

These model predictions become features for the final ML ensemble.
This is a "stacking" approach where NLP model outputs are used as inputs.

Input: Stage 2.2 output (spaCy features + review_text)
Output: Parquet files with NB_prob and svm_pred columns added

Note: Uses scikit-learn (not Spark NLP) for simplicity and portability.
Pipeline: 2.1 → 2.2 → 2.3 (this stage) → 2.4 → 2.5
"""

import logging
import sys
import joblib
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import PathConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(input_dir: Path, split: str) -> pd.DataFrame:
    """Load a data split from Stage 2.2 output."""
    path = input_dir / f"{split}.parquet"
    df = pd.read_parquet(path)
    logger.info(f"Loaded {split}: {len(df):,} rows, {len(df.columns)} cols")
    return df


def train_tfidf_models(train_df: pd.DataFrame, target_col: str = "T2_CLS_ufc_>0"):
    """Train TF-IDF vectorizer and classification models.

    Args:
        train_df: Training DataFrame with review_text and target
        target_col: Binary target column name

    Returns:
        Tuple of (vectorizer, nb_model, svm_model)
    """
    logger.info("Training TF-IDF vectorizer...")

    # TF-IDF vectorizer with similar settings to original
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
    )

    # Fit vectorizer and transform train data
    X_train = vectorizer.fit_transform(train_df["review_text"].fillna(""))
    y_train = train_df[target_col].astype(int)

    logger.info(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,}")
    logger.info(f"Feature matrix shape: {X_train.shape}")

    # Train Naive Bayes (gives probability directly)
    logger.info("Training Naive Bayes classifier...")
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train, y_train)

    # Train SVM with calibration (to get probabilities)
    logger.info("Training SVM classifier...")
    svm_base = LinearSVC(max_iter=10000, random_state=42)
    svm_model = CalibratedClassifierCV(svm_base, cv=3)
    svm_model.fit(X_train, y_train)

    return vectorizer, nb_model, svm_model


def add_model_predictions(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    nb_model: MultinomialNB,
    svm_model: CalibratedClassifierCV,
) -> pd.DataFrame:
    """Add NB_prob and svm_pred columns to DataFrame.

    Args:
        df: DataFrame with review_text
        vectorizer: Fitted TF-IDF vectorizer
        nb_model: Fitted Naive Bayes model
        svm_model: Fitted SVM model

    Returns:
        DataFrame with prediction columns added
    """
    # Transform text to TF-IDF features
    X = vectorizer.transform(df["review_text"].fillna(""))

    # Get Naive Bayes probability of positive class
    nb_probs = nb_model.predict_proba(X)[:, 1]
    df["NB_prob"] = np.round(nb_probs, 4)

    # Get SVM prediction score (using calibrated probabilities)
    svm_probs = svm_model.predict_proba(X)[:, 1]
    df["svm_pred"] = np.round(svm_probs, 4)

    return df


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Stage 2.3: TF-IDF Vectorization and ML Models")
    logger.info("=" * 60)

    # Setup paths
    input_dir = PathConfig.get_nlp_spacy_dir()  # From 2.2 (spaCy features)
    output_dir = PathConfig.get_nlp_tfidf_dir()
    model_dir = PathConfig.get_nlp_models_dir()

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model directory: {model_dir}")

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_df = load_data(input_dir, "train")

    # Train models
    vectorizer, nb_model, svm_model = train_tfidf_models(train_df)

    # Save models for later use
    logger.info("Saving models...")
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")
    joblib.dump(nb_model, model_dir / "naive_bayes_tfidf.joblib")
    joblib.dump(svm_model, model_dir / "svm_tfidf.joblib")

    # Process each split
    total_records = 0
    for split in ["train", "test", "holdout"]:
        logger.info(f"Processing {split} split...")

        # Load data
        df = load_data(input_dir, split)

        # Add predictions
        df = add_model_predictions(df, vectorizer, nb_model, svm_model)

        # Save output
        output_path = output_dir / f"{split}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"  Saved {len(df):,} rows to {output_path}")

        total_records += len(df)

    logger.info("=" * 60)
    logger.info(f"Completed! Processed {total_records:,} total records")
    logger.info(f"Features added: NB_prob, svm_pred")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
