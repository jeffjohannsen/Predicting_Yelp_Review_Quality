"""
Stage 2.4: FastText Text Classification

Trains a FastText supervised classifier to generate prediction features:
- ft_prob: FastText probability of positive class (review quality)

FastText uses subword information which helps with misspellings and rare words.
This model prediction becomes a feature for the final ML ensemble.

Input: Stage 2.3 output (TF-IDF features + review_text)
Pipeline: 2.1 → 2.2 → 2.3 → 2.4 (this stage) → 2.5
Output: Parquet files with ft_prob column added

Note: The original pipeline also included Universal Sentence Encoder (USE)
embeddings via Spark NLP. This simplified version uses only FastText.
USE could be added in Phase 2 with sentence-transformers.
"""

import csv
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import fasttext
from gensim.utils import simple_preprocess

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import PathConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress FastText warnings
fasttext.FastText.eprint = lambda x: None


def preprocess_text(text: str) -> str:
    """Preprocess text for FastText using gensim simple_preprocess."""
    if not text:
        return ""
    return " ".join(simple_preprocess(text))


def create_fasttext_file(df: pd.DataFrame, filepath: Path, target_col: str = "T2_CLS_ufc_>0"):
    """Create FastText training file format.

    FastText expects: __label__<label> <text>

    Args:
        df: DataFrame with review_text and target column
        filepath: Output file path
        target_col: Target column name (binary)
    """
    # Preprocess text
    texts = df["review_text"].fillna("").apply(preprocess_text)

    # Create labels
    labels = df[target_col].apply(lambda x: f"__label__{bool(x)}")

    # Combine and save
    with open(filepath, "w") as f:
        for label, text in zip(labels, texts):
            f.write(f"{label} {text}\n")

    logger.info(f"Created FastText file: {filepath}")


def get_positive_prob(prediction: tuple) -> float:
    """Extract probability of positive class from FastText prediction.

    Args:
        prediction: Tuple of (labels, probabilities) from model.predict()

    Returns:
        Probability of positive class (True)
    """
    label, prob = prediction[0][0], prediction[1][0]
    if label == "__label__True":
        return round(prob, 4)
    else:
        return round(1 - prob, 4)


def train_fasttext_model(train_file: Path, model_path: Path):
    """Train FastText supervised classifier.

    Args:
        train_file: Path to FastText format training file
        model_path: Path to save trained model

    Returns:
        Trained FastText model
    """
    logger.info("Training FastText model...")

    model = fasttext.train_supervised(
        input=str(train_file),
        loss="hs",  # Hierarchical softmax (faster for binary)
        epoch=25,
        lr=0.5,
        wordNgrams=2,
        dim=100,
    )

    # Save model
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")

    return model


def safe_predict(model, text: str) -> tuple:
    """Safely predict with FastText, handling numpy 2.0 issues."""
    try:
        return model.predict(text)
    except ValueError:
        # Numpy 2.0 compatibility issue - get raw prediction
        labels, probs = model.predict(text, k=1)
        return (labels, [probs[0]] if isinstance(probs, (list, np.ndarray)) else [probs])


def add_fasttext_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    """Add FastText prediction column to DataFrame.

    Args:
        df: DataFrame with review_text
        model: Trained FastText model

    Returns:
        DataFrame with ft_prob column added
    """
    # Preprocess texts
    texts = df["review_text"].fillna("").apply(preprocess_text).tolist()

    # Get predictions with error handling
    ft_probs = []
    for text in texts:
        try:
            labels, probs = model.predict(text)
            label = labels[0]
            prob = probs[0] if hasattr(probs, '__getitem__') else probs
            if label == "__label__True":
                ft_probs.append(round(float(prob), 4))
            else:
                ft_probs.append(round(1 - float(prob), 4))
        except Exception as e:
            ft_probs.append(0.5)  # Default to neutral

    df["ft_prob"] = ft_probs
    return df


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Stage 2.4: FastText Text Classification")
    logger.info("=" * 60)

    # Setup paths
    input_dir = PathConfig.get_nlp_tfidf_dir()  # From 2.3 (TF-IDF features)
    output_dir = PathConfig.get_nlp_embeddings_dir()
    model_dir = PathConfig.get_nlp_models_dir()

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model directory: {model_dir}")

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_df = pd.read_parquet(input_dir / "train.parquet")
    logger.info(f"Loaded train: {len(train_df):,} rows")

    # Create temporary FastText training file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        train_file = Path(f.name)

    create_fasttext_file(train_df, train_file)

    # Train model
    model_path = model_dir / "fasttext_model.bin"
    model = train_fasttext_model(train_file, model_path)

    # Test model
    test_results = model.test(str(train_file))
    logger.info(f"Training accuracy: {test_results[1]:.4f}")

    # Clean up temp file
    train_file.unlink()

    # Process each split
    total_records = 0
    for split in ["train", "test", "holdout"]:
        logger.info(f"Processing {split} split...")

        # Load data
        df = pd.read_parquet(input_dir / f"{split}.parquet")

        # Add predictions
        df = add_fasttext_predictions(df, model)

        # Save output
        output_path = output_dir / f"{split}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"  Saved {len(df):,} rows to {output_path}")

        total_records += len(df)

    logger.info("=" * 60)
    logger.info(f"Completed! Processed {total_records:,} total records")
    logger.info(f"Features added: ft_prob")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
