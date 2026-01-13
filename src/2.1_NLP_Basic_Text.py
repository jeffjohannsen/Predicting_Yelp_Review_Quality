"""
Stage 2.1: NLP Basic Text Processing

Extracts basic text features from review text:
- Word count, character count, average word length
- Number counts, uppercase counts, hashtag/mention counts
- Sentence count, lexicon count, syllable count
- Flesch-Kincaid grade level (readability)
- Sentiment polarity and subjectivity (TextBlob)

Input: ETL output (train_text.parquet, test_text.parquet, holdout_text.parquet)
Output: Parquet files with text features added
"""

import logging
import sys
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, IntegerType

import textstat
from textblob import TextBlob

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import PathConfig, SparkConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_spark_session():
    """Create and configure Spark session."""
    return (
        SparkSession.builder
        .appName("NLP_2.1_Basic_Text_Features")
        .master(SparkConfig.SPARK_MASTER)
        .config("spark.driver.memory", "8g")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def avg_word_length(text: str) -> float:
    """Calculate average word length in text."""
    if not text:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    return sum(len(word) for word in words) / len(words)


def count_numbers(text: str) -> int:
    """Count words that are digits."""
    if not text:
        return 0
    return len([w for w in text.split() if w.isdigit()])


def count_uppercase(text: str) -> int:
    """Count fully uppercase words."""
    if not text:
        return 0
    return len([w for w in text.split() if w.isupper()])


def count_hashtags_mentions(text: str) -> int:
    """Count words starting with # or @."""
    if not text:
        return 0
    return len([w for w in text.split() if w.startswith("#") or w.startswith("@")])


def get_polarity(text: str) -> float:
    """Get sentiment polarity from TextBlob."""
    if not text:
        return 0.0
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0


def get_subjectivity(text: str) -> float:
    """Get sentiment subjectivity from TextBlob."""
    if not text:
        return 0.0
    try:
        return TextBlob(text).sentiment.subjectivity
    except Exception:
        return 0.0


def safe_sentence_count(text: str) -> int:
    """Safely count sentences."""
    if not text:
        return 0
    try:
        return textstat.sentence_count(text)
    except Exception:
        return 0


def safe_lexicon_count(text: str) -> int:
    """Safely count lexicon items."""
    if not text:
        return 0
    try:
        return textstat.lexicon_count(text)
    except Exception:
        return 0


def safe_syllable_count(text: str) -> int:
    """Safely count syllables."""
    if not text:
        return 0
    try:
        return textstat.syllable_count(text)
    except Exception:
        return 0


def safe_flesch_kincaid(text: str) -> float:
    """Safely calculate Flesch-Kincaid grade level."""
    if not text:
        return 0.0
    try:
        return float(textstat.flesch_kincaid_grade(text))
    except Exception:
        return 0.0


def add_text_features(df):
    """Add all text features to a DataFrame.

    Args:
        df: Spark DataFrame with 'review_text' column

    Returns:
        DataFrame with text features added
    """
    # Register UDFs
    word_count_udf = F.udf(lambda x: len(str(x).split()) if x else 0, IntegerType())
    char_count_udf = F.udf(lambda x: len(x) if x else 0, IntegerType())
    avg_word_udf = F.udf(avg_word_length, FloatType())
    num_count_udf = F.udf(count_numbers, IntegerType())
    upper_count_udf = F.udf(count_uppercase, IntegerType())
    hashtag_udf = F.udf(count_hashtags_mentions, IntegerType())
    sentence_udf = F.udf(safe_sentence_count, IntegerType())
    lexicon_udf = F.udf(safe_lexicon_count, IntegerType())
    syllable_udf = F.udf(safe_syllable_count, IntegerType())
    grade_udf = F.udf(safe_flesch_kincaid, FloatType())
    polarity_udf = F.udf(get_polarity, FloatType())
    subjectivity_udf = F.udf(get_subjectivity, FloatType())

    # Apply all transformations
    return (
        df
        .withColumn("word_count", word_count_udf("review_text"))
        .withColumn("character_count", char_count_udf("review_text"))
        .withColumn("avg_word_length", avg_word_udf("review_text"))
        .withColumn("num_count", num_count_udf("review_text"))
        .withColumn("uppercase_count", upper_count_udf("review_text"))
        .withColumn("hashtag_mention_count", hashtag_udf("review_text"))
        .withColumn("sentence_count", sentence_udf("review_text"))
        .withColumn("lexicon_count", lexicon_udf("review_text"))
        .withColumn("syllable_count", syllable_udf("review_text"))
        .withColumn("grade_level", grade_udf("review_text"))
        .withColumn("polarity", polarity_udf("review_text"))
        .withColumn("subjectivity", subjectivity_udf("review_text"))
    )


def process_split(spark, split_name: str, input_dir: Path, output_dir: Path):
    """Process a single data split (train/test/holdout).

    Args:
        spark: SparkSession
        split_name: Name of split ('train', 'test', 'holdout')
        input_dir: Path to ETL output directory
        output_dir: Path to NLP output directory
    """
    input_path = input_dir / f"{split_name}_text.parquet"
    output_path = output_dir / f"{split_name}.parquet"

    logger.info(f"Processing {split_name} split...")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output: {output_path}")

    # Read data
    df = spark.read.parquet(str(input_path))
    record_count = df.count()
    logger.info(f"  Loaded {record_count:,} records")

    # Add text features
    df_features = add_text_features(df)

    # Write output
    df_features.write.mode("overwrite").parquet(str(output_path))
    logger.info(f"  Wrote {split_name} features to {output_path}")

    return record_count


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Stage 2.1: NLP Basic Text Processing")
    logger.info("=" * 60)

    # Setup paths
    input_dir = PathConfig.get_etl_output_dir()
    output_dir = PathConfig.get_nlp_basic_dir()

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Spark session
    spark = create_spark_session()
    logger.info(f"Spark session created: {spark.sparkContext.master}")

    try:
        # Process each split
        total_records = 0
        for split in ["train", "test", "holdout"]:
            count = process_split(spark, split, input_dir, output_dir)
            total_records += count

        logger.info("=" * 60)
        logger.info(f"Completed! Processed {total_records:,} total records")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 60)

    finally:
        spark.stop()
        logger.info("Spark session stopped")


if __name__ == "__main__":
    main()
