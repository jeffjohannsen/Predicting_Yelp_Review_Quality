"""
Stage 3: Combine Features and Export to Model-Ready Format

Takes Stage 2 NLP output and prepares it for ML training:
- Drops non-feature columns (review_text)
- Standardizes column names for consistency with original pipeline
- Applies memory-efficient dtypes (int16, float32)
- Exports to CSV format for sklearn compatibility

Input: Stage 2.5 output (LDA features - final NLP stage)
Output: Model-ready CSV files in data/model_ready/

Pipeline: Stage 1 (ETL) → Stage 2 (NLP 2.1-2.5) → Stage 3 (this) → Stage 4 (ML)
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import PathConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Column name mapping: new names -> old names (for backward compatibility)
# The original pipeline used different naming conventions
COLUMN_RENAME_MAP = {
    # Basic text features - rename for consistency
    "word_count": "word_cnt",
    "character_count": "character_cnt",
    "num_count": "num_cnt",
    "uppercase_count": "uppercase_cnt",
    "hashtag_mention_count": "hashtag_cnt",
    "sentence_count": "sentence_cnt",
    "lexicon_count": "lexicon_cnt",
    "syllable_count": "syllable_cnt",
    "avg_word_length": "avg_word_len",
    # spaCy features - count suffix
    "token_count": "token_cnt",
    "stopword_count": "stopword_cnt",
    "ent_count": "ent_cnt",
}

# Build percentage column renames dynamically
# Current: _perc -> Old: _pct
def build_perc_rename_map(columns: list) -> dict:
    """Build rename map for percentage columns (_perc -> _pct)."""
    rename_map = {}
    for col in columns:
        if col.endswith("_perc"):
            new_name = col.replace("_perc", "_pct")
            rename_map[col] = new_name
    return rename_map


# Build count column renames dynamically
# Current: _count -> Old: _cnt (for pos_, dep_, ent_ columns)
def build_count_rename_map(columns: list) -> dict:
    """Build rename map for count columns (_count -> _cnt)."""
    rename_map = {}
    for col in columns:
        if col.endswith("_count") and (
            col.startswith("pos_") or col.startswith("dep_") or col.startswith("ent_")
        ):
            new_name = col.replace("_count", "_cnt")
            rename_map[col] = new_name
    return rename_map


# Columns to drop (not needed for ML)
COLUMNS_TO_DROP = [
    "review_text",  # Raw text not needed for ML features
]

# Target columns (keep these separate for clarity)
TARGET_COLUMNS = [
    "T1_REG_review_total_ufc",
    "T2_CLS_ufc_>0",
    "T3_CLS_ufc_level",
    "T4_REG_ufc_TD",
    "T5_CLS_ufc_level_TD",
    "T6_REG_ufc_TDBD",
]

# ID columns
ID_COLUMNS = ["review_id"]


def get_dtype_map(columns: list) -> dict:
    """Build dtype map for memory-efficient storage.

    Args:
        columns: List of column names

    Returns:
        Dictionary mapping column names to dtypes
    """
    dtype_map = {}

    for col in columns:
        # Skip ID columns (keep as string/object)
        if col in ID_COLUMNS:
            continue

        # Target columns
        if col in TARGET_COLUMNS:
            if "CLS" in col or "bool" in col.lower():
                dtype_map[col] = "bool"
            else:
                dtype_map[col] = "float32"
            continue

        # Count columns -> int16
        if "_cnt" in col or "_count" in col or col.endswith("_cnt"):
            dtype_map[col] = "int16"
            continue

        # Percentage/probability columns -> float32
        if "_pct" in col or "_perc" in col or "_prob" in col or col in [
            "polarity", "subjectivity", "grade_level", "avg_word_len",
            "lda_t1", "lda_t2", "lda_t3", "lda_t4", "lda_t5"
        ]:
            dtype_map[col] = "float32"
            continue

        # Integer features
        if col in ["review_stars", "word_cnt", "character_cnt", "num_cnt",
                   "uppercase_cnt", "hashtag_cnt", "sentence_cnt", "lexicon_cnt",
                   "syllable_cnt", "token_cnt", "stopword_cnt", "ent_cnt"]:
            dtype_map[col] = "int16"
            continue

        # Prediction columns
        if col in ["NB_prob", "svm_pred", "ft_prob"]:
            dtype_map[col] = "float32"
            continue

    return dtype_map


def process_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Process a single data split.

    Args:
        df: Input DataFrame from Stage 2
        split_name: Name of split for logging

    Returns:
        Processed DataFrame ready for ML
    """
    logger.info(f"Processing {split_name}: {len(df):,} rows, {len(df.columns)} columns")

    # Drop non-feature columns
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"  Dropped columns: {cols_to_drop}")

    # Build rename maps
    rename_map = COLUMN_RENAME_MAP.copy()
    rename_map.update(build_perc_rename_map(df.columns.tolist()))
    rename_map.update(build_count_rename_map(df.columns.tolist()))

    # Apply renames (only for columns that exist)
    actual_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    if actual_renames:
        df = df.rename(columns=actual_renames)
        logger.info(f"  Renamed {len(actual_renames)} columns")

    # Apply memory-efficient dtypes
    dtype_map = get_dtype_map(df.columns.tolist())
    for col, dtype in dtype_map.items():
        if col in df.columns:
            try:
                if dtype == "bool":
                    df[col] = df[col].astype(bool)
                elif dtype == "int16":
                    # Handle potential overflow by clipping
                    df[col] = df[col].clip(-32768, 32767).astype("int16")
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                logger.warning(f"  Could not convert {col} to {dtype}: {e}")

    logger.info(f"  Final shape: {df.shape}")
    return df


def save_feature_names(df: pd.DataFrame, output_dir: Path):
    """Save feature names to text file.

    Args:
        df: Processed DataFrame
        output_dir: Output directory
    """
    # Get feature columns (exclude ID and target columns)
    feature_cols = [c for c in df.columns
                   if c not in ID_COLUMNS and c not in TARGET_COLUMNS]

    feature_file = output_dir / "feature_names.txt"
    with open(feature_file, "w") as f:
        f.write(f"# Feature names for Yelp Review Quality Prediction\n")
        f.write(f"# Total features: {len(feature_cols)}\n")
        f.write(f"# Generated by Stage 3: Combine Features\n\n")
        for col in sorted(feature_cols):
            f.write(f"{col}\n")

    logger.info(f"Saved {len(feature_cols)} feature names to {feature_file}")


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Stage 3: Combine Features and Export to Model-Ready Format")
    logger.info("=" * 60)

    # Setup paths
    input_dir = PathConfig.get_nlp_lda_dir()  # Final Stage 2 output
    output_dir = PathConfig.get_model_ready_dir()

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    total_records = 0
    for split in ["train", "test", "holdout"]:
        input_path = input_dir / f"{split}.parquet"

        if not input_path.exists():
            logger.warning(f"Input file not found: {input_path}")
            continue

        # Load data
        df = pd.read_parquet(input_path)

        # Process
        df = process_split(df, split)

        # Save as CSV
        output_path = output_dir / f"{split}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"  Saved to {output_path}")

        # Save feature names (from train split)
        if split == "train":
            save_feature_names(df, output_dir)

        total_records += len(df)

    # Summary
    logger.info("=" * 60)
    logger.info(f"Completed! Processed {total_records:,} total records")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
