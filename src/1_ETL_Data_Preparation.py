"""
ETL Pipeline for Yelp Review Quality Prediction
Stage 1: Load Parquet data, apply time discounting, create train/test splits

This script:
1. Loads preprocessed Parquet files (from data/parquet_2021/)
2. Applies time discounting to targets and user/business features
3. Joins review, user, business, and checkin data
4. Creates train/test/holdout splits
5. Separates text and non-text features
6. Saves processed data to Parquet (data/processed/01_etl_output/)

Key Changes from v1:
- Input: Parquet files instead of JSON
- Added: Complete time discounting for all features (2020-era methodology)
- Output: Parquet instead of PostgreSQL
- Added: Logging and error handling
- Removed: Hardcoded paths (uses config.py)
- Removed: PostgreSQL JDBC dependencies

PERFORMANCE NOTE:
- Uses Python UDFs for time discounting (compatible but slower)
- Processing 8M reviews may take 2-4 hours on typical hardware
- For production, consider Pandas UDFs (10x faster) or pre-computing discounts
- This implementation prioritizes code clarity over raw performance

METHODOLOGY NOTE:
- This represents a 2020-era academic approach with extensive feature engineering
- Time discounting is a workaround for using 5-year-old dataset
- Modern production systems would use recent data + transformer embeddings
- Phase 2 will implement SOTA approach with BERT/sentence-transformers

Author: Jeff (refactored from original v1)
Date: December 2025
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import logging
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    FloatType,
    IntegerType,
    StringType,
    TimestampType,
)

# Import project utilities
from config import PathConfig
from utils.spark_helpers import create_spark_session, stop_spark_session
from utils.time_discount import TimeDiscountCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_parquet_data(spark, sample_size=None):
    """
    Load all Parquet files from data/parquet_2021/ directory.

    Args:
        spark: SparkSession
        sample_size: Optional limit for review data (for testing). If None, loads all.

    Returns:
        Tuple of (df_business, df_checkin, df_review, df_user)
    """
    logger.info("Loading Parquet data files...")

    try:
        df_business = spark.read.parquet(str(PathConfig.get_parquet_business()))
        logger.info(f"Loaded business data: {df_business.count():,} records")

        df_checkin = spark.read.parquet(str(PathConfig.get_parquet_checkin()))
        logger.info(f"Loaded checkin data: {df_checkin.count():,} records")

        df_review = spark.read.parquet(str(PathConfig.get_parquet_review()))
        if sample_size:
            df_review = df_review.limit(sample_size)
            logger.info(f"Loaded review data (SAMPLE: {sample_size:,}): {df_review.count():,} records")
        else:
            logger.info(f"Loaded review data: {df_review.count():,} records")

        df_user = spark.read.parquet(str(PathConfig.get_parquet_user()))
        logger.info(f"Loaded user data: {df_user.count():,} records")

        return df_business, df_checkin, df_review, df_user

    except Exception as e:
        logger.error(f"Error loading Parquet data: {e}")
        raise


def process_checkin_data(df_checkin, spark):
    """
    Process checkin data to extract count and min/max dates.

    Original used comma-split timestamp strings. Parquet has actual timestamp array.

    Args:
        df_checkin: Raw checkin DataFrame
        spark: SparkSession

    Returns:
        Processed DataFrame with business_id, num_checkins, checkin_min, checkin_max
    """
    logger.info("Processing checkin data...")

    df_checkin.createOrReplaceTempView("df_checkin")

    df_checkin_final = spark.sql(
        """
        SELECT business_id,
            size(date_array) AS num_checkins,
            array_min(date_array) AS checkin_min,
            array_max(date_array) AS checkin_max
        FROM df_checkin
    """
    )

    logger.info(f"Checkin data processed: {df_checkin_final.count():,} businesses")
    return df_checkin_final


def process_user_data(df_user, spark):
    """
    Process user data to aggregate features and handle elite years.

    Transforms:
    - Split elite years into array
    - Calculate total compliments
    - Calculate total UFC (useful + funny + cool)
    - Extract friend count from friends string

    Args:
        df_user: Raw user DataFrame
        spark: SparkSession

    Returns:
        Processed DataFrame with user features
    """
    logger.info("Processing user data...")

    df_user.createOrReplaceTempView("df_user")

    # First pass: aggregate compliments and split arrays
    df_user_1 = spark.sql(
        """
        SELECT user_id,
            yelping_since,
            CASE 
                WHEN elite = 'None' OR elite IS NULL OR elite = '' THEN array()
                ELSE split(replace(elite, '20,20', '2020'), ',')
            END AS elite_array,
            average_stars,
            review_count,
            fans,
            CASE 
                WHEN friends = 'None' OR friends IS NULL OR friends = '' THEN 0
                ELSE size(split(friends, ','))
            END AS friend_count,
            (compliment_cool + compliment_cute + compliment_funny
            + compliment_hot + compliment_list + compliment_more
            + compliment_note + compliment_photos + compliment_plain
            + compliment_profile + compliment_writer) AS compliments,
            (cool + funny + useful) AS ufc_count
        FROM df_user
    """
    )

    df_user_1.createOrReplaceTempView("df_user_1")

    # Second pass: calculate elite metrics
    df_user_final = spark.sql(
        """
        SELECT user_id,
            yelping_since,
            elite_array,
            average_stars,
            review_count,
            fans,
            friend_count,
            compliments,
            ufc_count,
            size(elite_array) AS elite_count,
            CASE
                WHEN size(elite_array) = 0 THEN 0
                ELSE int(array_min(elite_array))
            END AS elite_min,
            CASE
                WHEN size(elite_array) = 0 THEN 0
                ELSE int(array_max(elite_array))
            END AS elite_max
        FROM df_user_1
    """
    )

    logger.info(f"User data processed: {df_user_final.count():,} users")
    return df_user_final


def process_business_data(df_business, spark):
    """
    Process business data to select relevant features.

    Args:
        df_business: Raw business DataFrame
        spark: SparkSession

    Returns:
        Processed DataFrame with business features
    """
    logger.info("Processing business data...")

    df_business.createOrReplaceTempView("df_business")

    df_business_final = spark.sql(
        """
        SELECT business_id,
            latitude,
            longitude,
            postal_code,
            state,
            stars,
            review_count
        FROM df_business
    """
    )

    logger.info(f"Business data processed: {df_business_final.count():,} businesses")
    return df_business_final


def process_review_data(df_review, spark):
    """
    Process review data to create initial targets and extract vote counts.

    Note: This creates the raw vote counts. Time discounting happens later
    after joining with user data (need review_date for discounting).

    Args:
        df_review: Raw review DataFrame
        spark: SparkSession

    Returns:
        Processed DataFrame with review features and raw vote counts
    """
    logger.info("Processing review data...")

    df_review.createOrReplaceTempView("df_review")

    df_review_final = spark.sql(
        """
        SELECT review_id,
            date AS review_date,
            user_id,
            business_id,
            stars,
            text,
            useful,
            funny,
            cool,
            (cool + funny + useful) AS ufc_total
        FROM df_review
    """
    )

    logger.info(f"Review data processed: {df_review_final.count():,} reviews")
    return df_review_final


def join_all_data(df_review, df_user, df_business, df_checkin, spark):
    """
    Join all data tables together.

    Performs left joins from reviews to users, businesses, and checkins.
    This creates the complete feature set for each review.

    Args:
        df_review: Processed review DataFrame
        df_user: Processed user DataFrame
        df_business: Processed business DataFrame
        df_checkin: Processed checkin DataFrame
        spark: SparkSession

    Returns:
        Joined DataFrame with all features
    """
    logger.info("Joining all data tables...")

    df_checkin.createOrReplaceTempView("df_checkin_final")
    df_user.createOrReplaceTempView("df_user_final")
    df_business.createOrReplaceTempView("df_business_final")
    df_review.createOrReplaceTempView("df_review_final")

    all_data = spark.sql(
        """
        SELECT r.review_id,
            r.user_id,
            r.business_id,
            b.latitude AS biz_latitude,
            b.longitude AS biz_longitude,
            b.postal_code AS biz_postal_code,
            b.state AS biz_state,
            b.stars AS biz_avg_stars,
            b.review_count AS biz_review_count,
            c.num_checkins AS biz_checkin_count,
            c.checkin_min AS biz_min_checkin_date,
            c.checkin_max AS biz_max_checkin_date,
            u.yelping_since AS user_yelping_since,
            u.elite_array AS user_elite_array,
            u.elite_count AS user_elite_count,
            u.elite_min AS user_elite_min,
            u.elite_max AS user_elite_max,
            u.average_stars AS user_avg_stars,
            u.review_count AS user_review_count,
            u.fans AS user_fan_count,
            u.friend_count AS user_friend_count,
            u.compliments AS user_compliment_count,
            u.ufc_count AS user_ufc_count,
            r.review_date AS review_date,
            r.stars AS review_stars,
            r.text AS review_text,
            r.useful AS review_useful,
            r.funny AS review_funny,
            r.cool AS review_cool,
            r.ufc_total AS review_ufc_total
        FROM df_review_final AS r
        LEFT JOIN df_user_final AS u
        ON r.user_id = u.user_id
        LEFT JOIN df_business_final AS b
        ON r.business_id = b.business_id
        LEFT JOIN df_checkin_final AS c
        ON r.business_id = c.business_id
    """
    )

    logger.info(f"All data joined: {all_data.count():,} records")
    return all_data


def apply_time_discounting(df, spark):
    """
    Apply time discounting to all features using Spark UDFs.

    This is the critical step that adjusts feature values to represent
    their approximate values at review creation time rather than at
    dataset release time (2020-03-25).

    Creates:
    - 6 target variations (T1-T6)
    - Time-discounted user features (5 features)
    - Time-discounted business features (2 features)
    - Time-discounted elite features (2 features)

    Args:
        df: Joined DataFrame with all raw features
        spark: SparkSession

    Returns:
        DataFrame with time-discounted features added
    """
    logger.info("Applying time discounting to features...")

    # Register UDFs for time discounting
    # PERFORMANCE NOTE: Python UDFs serialize data between Python/JVM for each row
    # This is slow but maintains code clarity and matches original methodology
    # For 10x speedup, refactor to Pandas UDFs (see backlog in TODO.md)
    # Current implementation: ~2-4 hours for 8M reviews (acceptable for batch processing)
    # 
    # IMPORTANT: Create calculator inside each UDF to avoid serialization issues
    # Workers need to be able to import utils.time_discount module

    # Target time discount UDF
    @F.udf(returnType=FloatType())
    def udf_target_td(ufc_total, review_date):
        from utils.time_discount import TimeDiscountCalculator
        calc = TimeDiscountCalculator()
        if ufc_total is None or review_date is None:
            return 0.0
        return float(calc.target_time_discount(int(ufc_total), review_date))

    # User time discount UDF
    @F.udf(returnType=FloatType())
    def udf_user_td(count_val, user_since, review_date):
        from utils.time_discount import TimeDiscountCalculator
        calc = TimeDiscountCalculator()
        if count_val is None or user_since is None or review_date is None:
            return 0.0
        return float(calc.user_time_discount(float(count_val), user_since, review_date))

    # Business time discount UDF
    @F.udf(returnType=FloatType())
    def udf_business_td(count_val, review_date):
        from utils.time_discount import TimeDiscountCalculator
        calc = TimeDiscountCalculator()
        if count_val is None or review_date is None:
            return 0.0
        return float(calc.business_time_discount(float(count_val), review_date))

    # Elite count time discount UDF
    @F.udf(returnType=IntegerType())
    def udf_elite_count_td(elite_array, review_date):
        from utils.time_discount import TimeDiscountCalculator
        calc = TimeDiscountCalculator()
        if elite_array is None or review_date is None or len(elite_array) == 0:
            return 0
        # Convert array to comma-separated string for function
        elite_str = ",".join(map(str, elite_array))
        return calc.count_elite_td(elite_str, review_date)

    # Years since elite UDF
    @F.udf(returnType=IntegerType())
    def udf_years_since_elite_td(elite_array, review_date):
        from utils.time_discount import TimeDiscountCalculator
        calc = TimeDiscountCalculator()
        if elite_array is None or review_date is None or len(elite_array) == 0:
            return 100
        elite_str = ",".join(map(str, elite_array))
        return calc.years_since_elite_td(elite_str, review_date)

    # Usefulness level UDF
    @F.udf(returnType=StringType())
    def udf_usefulness_level(ufc_count):
        from utils.time_discount import TimeDiscountCalculator
        calc = TimeDiscountCalculator()
        if ufc_count is None:
            return "zero"
        return calc.usefulness_level(float(ufc_count))

    # Apply target transformations
    logger.info("  Creating target variables (T1-T6)...")

    # T1: Raw total (no discounting)
    df = df.withColumn("T1_REG_review_total_ufc", F.col("review_ufc_total"))

    # T2: Binary classification
    df = df.withColumn("T2_CLS_ufc_>0", F.col("review_ufc_total") > 0)

    # T3: Categorical level (no discounting)
    df = df.withColumn(
        "T3_CLS_ufc_level", udf_usefulness_level(F.col("review_ufc_total"))
    )

    # T4: Time discounted
    df = df.withColumn(
        "T4_REG_ufc_TD", udf_target_td(F.col("review_ufc_total"), F.col("review_date"))
    )

    # T5: Categorical level (time discounted)
    df = df.withColumn(
        "T5_CLS_ufc_level_TD", udf_usefulness_level(F.col("T4_REG_ufc_TD"))
    )

    # T6: Time + business popularity discounted
    df = df.withColumn(
        "T6_REG_ufc_TDBD",
        F.when(
            F.col("biz_review_count") > 0,
            F.col("T4_REG_ufc_TD") / F.col("biz_review_count"),
        ).otherwise(F.col("T4_REG_ufc_TD")),
    )

    # Apply user feature time discounting
    logger.info("  Time discounting user features...")
    user_features_to_discount = [
        ("user_ufc_count", "user_ufc_count_TD"),
        ("user_compliment_count", "user_compliment_count_TD"),
        ("user_review_count", "user_review_count_TD"),
        ("user_fan_count", "user_fan_count_TD"),
        ("user_friend_count", "user_friend_count_TD"),
    ]

    for orig_col, new_col in user_features_to_discount:
        df = df.withColumn(
            new_col,
            udf_user_td(
                F.col(orig_col), F.col("user_yelping_since"), F.col("review_date")
            ),
        )

    # Apply business feature time discounting
    logger.info("  Time discounting business features...")
    business_features_to_discount = [
        ("biz_review_count", "biz_review_count_TD"),
        ("biz_checkin_count", "biz_checkin_count_TD"),
    ]

    for orig_col, new_col in business_features_to_discount:
        # Handle nulls (businesses with no checkins)
        df = df.withColumn(
            new_col,
            F.when(
                F.col(orig_col).isNotNull(),
                udf_business_td(F.col(orig_col), F.col("review_date")),
            ).otherwise(0.0),
        )

    # Apply elite time discounting
    logger.info("  Time discounting elite features...")
    df = df.withColumn(
        "user_elite_count_TD",
        udf_elite_count_td(F.col("user_elite_array"), F.col("review_date")),
    )
    df = df.withColumn(
        "user_years_since_elite_TD",
        udf_years_since_elite_td(F.col("user_elite_array"), F.col("review_date")),
    )

    # Calculate non-time-discounted elite feature for comparison
    df = df.withColumn(
        "user_years_since_elite",
        F.when(
            F.col("user_elite_count") > 0, F.lit(2020) - F.col("user_elite_max")
        ).otherwise(100),
    )

    # Drop intermediate columns we don't need
    df = df.drop(
        "review_ufc_total",
        "user_elite_array",
        "review_useful",
        "review_funny",
        "review_cool",
    )

    logger.info(f"Time discounting complete: {df.columns.__len__()} total columns")
    return df


def create_data_splits(df, spark, seed=12345, temporal_split=True):
    """
    Split data into train, test, and holdout sets.

    TEMPORAL SPLIT (temporal_split=True, RECOMMENDED):
    - Sorts by review_date chronologically
    - Train: Oldest 70% (prevents future leakage)
    - Test: Next 15% (realistic future validation)
    - Holdout: Most recent 15% (final holdout on newest data)
    - Ensures no temporal leakage for time-series data

    RANDOM SPLIT (temporal_split=False, LEGACY):
    - Uses randomSplit (original methodology)
    - Results in 64/16/20 split due to nested splits
    - ⚠️ Warning: Can cause temporal leakage

    Args:
        df: DataFrame with all features
        spark: SparkSession
        seed: Random seed for reproducibility (default: 12345)
        temporal_split: If True, use temporal split; if False, use random (default: True)

    Returns:
        Tuple of (train_data, test_data, holdout_data)
    """
    logger.info("Creating train/test/holdout splits...")

    if temporal_split:
        logger.info("  Using TEMPORAL split (no future leakage)...")

        # Sort by review date
        df_sorted = df.orderBy("review_date")
        total_count = df_sorted.count()

        # Calculate split points
        train_cutoff = int(total_count * 0.70)
        test_cutoff = int(total_count * 0.85)  # 70% + 15%

        # Create row numbers
        from pyspark.sql.window import Window
        window = Window.orderBy("review_date")
        df_with_row_num = df_sorted.withColumn("row_num", F.row_number().over(window))

        # Split by row number (temporal order)
        train_data = df_with_row_num.filter(F.col("row_num") <= train_cutoff).drop("row_num")
        test_data = df_with_row_num.filter(
            (F.col("row_num") > train_cutoff) & (F.col("row_num") <= test_cutoff)
        ).drop("row_num")
        holdout_data = df_with_row_num.filter(F.col("row_num") > test_cutoff).drop("row_num")

        logger.info(f"  Train data: {train_data.count():,} records (oldest 70%)")
        logger.info(f"  Test data: {test_data.count():,} records (next 15%)")
        logger.info(f"  Holdout data: {holdout_data.count():,} records (newest 15%)")

        # Show date ranges
        train_dates = train_data.select(
            F.min("review_date").alias("min"), F.max("review_date").alias("max")
        ).collect()[0]
        test_dates = test_data.select(
            F.min("review_date").alias("min"), F.max("review_date").alias("max")
        ).collect()[0]
        holdout_dates = holdout_data.select(
            F.min("review_date").alias("min"), F.max("review_date").alias("max")
        ).collect()[0]

        logger.info(f"  Train dates: {train_dates['min']} to {train_dates['max']}")
        logger.info(f"  Test dates: {test_dates['min']} to {test_dates['max']}")
        logger.info(f"  Holdout dates: {holdout_dates['min']} to {holdout_dates['max']}")

    else:
        logger.info("  Using RANDOM split (legacy method)...")
        logger.warning("  ⚠️ Random split may cause temporal leakage!")

        # Original random split (legacy)
        working_data, holdout_data = df.randomSplit([0.8, 0.2], seed=seed)
        logger.info(f"  Working data: {working_data.count():,} records")
        logger.info(f"  Holdout data: {holdout_data.count():,} records")

        train_data, test_data = working_data.randomSplit([0.8, 0.2], seed=seed)
        logger.info(f"  Train data: {train_data.count():,} records")
        logger.info(f"  Test data: {test_data.count():,} records")

    return train_data, test_data, holdout_data


def separate_text_and_non_text_features(df, spark):
    """
    Separate data into text and non-text feature sets.

    Text data: Only review_id, review_stars, review_text, and targets
    Non-text data: All other features (user, business, metadata)

    Args:
        df: DataFrame with all features
        spark: SparkSession

    Returns:
        Tuple of (text_data, non_text_data)
    """
    df.createOrReplaceTempView("data")

    text_data = spark.sql(
        """
        SELECT review_id,
            review_stars,
            review_text,
            T1_REG_review_total_ufc,
            `T2_CLS_ufc_>0`,
            T3_CLS_ufc_level,
            T4_REG_ufc_TD,
            T5_CLS_ufc_level_TD,
            T6_REG_ufc_TDBD
        FROM data
    """
    )

    non_text_data = spark.sql(
        """
        SELECT review_id,
            user_id,
            business_id,
            review_stars,
            review_date,
            biz_avg_stars,
            biz_review_count,
            biz_review_count_TD,
            biz_checkin_count,
            biz_checkin_count_TD,
            biz_max_checkin_date,
            biz_min_checkin_date,
            biz_latitude,
            biz_longitude,
            biz_postal_code,
            biz_state,
            user_avg_stars,
            user_review_count,
            user_review_count_TD,
            user_friend_count,
            user_friend_count_TD,
            user_fan_count,
            user_fan_count_TD,
            user_compliment_count,
            user_compliment_count_TD,
            user_ufc_count,
            user_ufc_count_TD,
            user_elite_count,
            user_elite_count_TD,
            user_elite_max,
            user_elite_min,
            user_years_since_elite,
            user_years_since_elite_TD,
            user_yelping_since,
            T1_REG_review_total_ufc,
            `T2_CLS_ufc_>0`,
            T3_CLS_ufc_level,
            T4_REG_ufc_TD,
            T5_CLS_ufc_level_TD,
            T6_REG_ufc_TDBD
        FROM data
    """
    )

    return text_data, non_text_data


def save_to_parquet(df, output_dir, dataset_name, spark):
    """
    Save DataFrame to Parquet with compression.

    Args:
        df: DataFrame to save
        output_dir: Base output directory (e.g., "data/processed/01_etl_output")
        dataset_name: Name for this dataset (e.g., "train_text", "test_non_text")
        spark: SparkSession
    """
    output_path = f"{output_dir}/{dataset_name}.parquet"
    logger.info(f"  Saving {dataset_name} to {output_path}...")

    try:
        df.write.mode("overwrite").parquet(output_path)
        saved_count = spark.read.parquet(output_path).count()
        logger.info(f"  ✓ Saved {saved_count:,} records to {output_path}")
    except Exception as e:
        logger.error(f"  ✗ Error saving {dataset_name}: {e}")
        raise


def log_stage(stage_num, total_stages, stage_name, status="START"):
    """Log stage progress with clear visual markers."""
    if status == "START":
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"STAGE {stage_num}/{total_stages}: {stage_name}")
        logger.info("=" * 80)
    elif status == "DONE":
        logger.info(f"✓ STAGE {stage_num}/{total_stages} COMPLETE: {stage_name}")
        logger.info("-" * 80)


def main(sample_size=None):
    """
    Main ETL pipeline execution.

    Args:
        sample_size: Number of reviews to process. None = full dataset (~8.6M).
    """
    TOTAL_STAGES = 8
    start_time = datetime.now()
    stage_times = {}

    logger.info("")
    logger.info("=" * 80)
    logger.info("  YELP REVIEW ETL PIPELINE - Stage 1: Data Preparation")
    logger.info("=" * 80)
    logger.info(f"  Started at:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Sample size:  {sample_size if sample_size else 'FULL DATASET (~8.6M reviews)'}")
    logger.info(f"  Total stages: {TOTAL_STAGES}")
    logger.info("=" * 80)

    global spark  # Make spark accessible to UDFs

    try:
        # STAGE 1: Create Spark session
        log_stage(1, TOTAL_STAGES, "Initialize Spark Session")
        stage_start = datetime.now()

        src_path = str(Path(__file__).parent)
        import os
        current_pythonpath = os.environ.get('PYTHONPATH', '')
        pythonpath = f"{src_path}:{current_pythonpath}" if current_pythonpath else src_path

        # Memory-safe configuration for 4GB available RAM
        # Key settings:
        # - 1500m heap leaves ~500MB for OS overhead
        # - memory.fraction=0.4 reserves 60% for execution/shuffle (vs default 0.6)
        # - shuffle.partitions=200 creates smaller chunks that fit in memory
        # - offHeap disabled to avoid additional memory pressure
        # - shuffle.spill.compress=true reduces disk I/O when spilling
        spark = create_spark_session(
            app_name="Yelp_ETL_Data_Prep",
            memory_driver="1500m",
            memory_executor="1500m",
            log_level="WARN",
            **{
                "spark.executorEnv.PYTHONPATH": pythonpath,
                "spark.yarn.appMasterEnv.PYTHONPATH": pythonpath,
                # Memory management - aggressive spill-to-disk
                "spark.memory.fraction": "0.4",
                "spark.memory.storageFraction": "0.2",
                "spark.sql.shuffle.partitions": "200",
                # Disk spill settings
                "spark.shuffle.spill.compress": "true",
                "spark.shuffle.compress": "true",
                # Reduce parallelism to avoid memory spikes
                "spark.default.parallelism": "4",
                # Broadcast threshold - smaller to avoid large broadcasts
                "spark.sql.autoBroadcastJoinThreshold": "10m",
            }
        )
        stage_times[1] = datetime.now() - stage_start
        log_stage(1, TOTAL_STAGES, "Initialize Spark Session", "DONE")

        # STAGE 2: Load Parquet data
        log_stage(2, TOTAL_STAGES, "Load Parquet Data")
        stage_start = datetime.now()
        df_business, df_checkin, df_review, df_user = load_parquet_data(spark, sample_size=sample_size)
        stage_times[2] = datetime.now() - stage_start
        log_stage(2, TOTAL_STAGES, "Load Parquet Data", "DONE")

        # STAGE 3: Process individual tables
        log_stage(3, TOTAL_STAGES, "Process Individual Tables")
        stage_start = datetime.now()
        df_checkin_final = process_checkin_data(df_checkin, spark)
        df_user_final = process_user_data(df_user, spark)
        df_business_final = process_business_data(df_business, spark)
        df_review_final = process_review_data(df_review, spark)
        stage_times[3] = datetime.now() - stage_start
        log_stage(3, TOTAL_STAGES, "Process Individual Tables", "DONE")

        # STAGE 4: Join all data
        log_stage(4, TOTAL_STAGES, "Join All Data Tables")
        stage_start = datetime.now()
        all_data = join_all_data(
            df_review_final, df_user_final, df_business_final, df_checkin_final, spark
        )

        # Memory optimization: Write intermediate result to disk and re-read
        # This breaks the lineage and releases memory from previous DataFrames
        if sample_size is None:  # Only for full dataset runs
            logger.info("  Writing intermediate checkpoint to release memory...")
            checkpoint_path = str(PathConfig.get_etl_output_dir() / "_checkpoint_joined")
            all_data.write.mode("overwrite").parquet(checkpoint_path)
            # Clear cached data and re-read from checkpoint
            spark.catalog.clearCache()
            all_data = spark.read.parquet(checkpoint_path)
            logger.info("  Checkpoint written and reloaded")

        stage_times[4] = datetime.now() - stage_start
        log_stage(4, TOTAL_STAGES, "Join All Data Tables", "DONE")

        # STAGE 5: Apply time discounting (slowest stage)
        log_stage(5, TOTAL_STAGES, "Apply Time Discounting (SLOW - Python UDFs)")
        stage_start = datetime.now()
        all_data_discounted = apply_time_discounting(all_data, spark)

        # Memory optimization: Checkpoint after time discounting (heaviest operation)
        if sample_size is None:  # Only for full dataset runs
            logger.info("  Writing post-discounting checkpoint to release memory...")
            checkpoint_path = str(PathConfig.get_etl_output_dir() / "_checkpoint_discounted")
            all_data_discounted.write.mode("overwrite").parquet(checkpoint_path)
            spark.catalog.clearCache()
            all_data_discounted = spark.read.parquet(checkpoint_path)
            logger.info("  Checkpoint written and reloaded")

        stage_times[5] = datetime.now() - stage_start
        log_stage(5, TOTAL_STAGES, "Apply Time Discounting", "DONE")

        # STAGE 6: Create train/test/holdout splits
        log_stage(6, TOTAL_STAGES, "Create Train/Test/Holdout Splits")
        stage_start = datetime.now()
        train_data, test_data, holdout_data = create_data_splits(
            all_data_discounted, spark
        )
        stage_times[6] = datetime.now() - stage_start
        log_stage(6, TOTAL_STAGES, "Create Train/Test/Holdout Splits", "DONE")

        # STAGE 7: Separate text and non-text features
        log_stage(7, TOTAL_STAGES, "Separate Text and Non-Text Features")
        stage_start = datetime.now()

        train_text, train_non_text = separate_text_and_non_text_features(train_data, spark)
        logger.info(f"  Train - Text: {train_text.count():,}, Non-text: {train_non_text.count():,}")

        test_text, test_non_text = separate_text_and_non_text_features(test_data, spark)
        logger.info(f"  Test - Text: {test_text.count():,}, Non-text: {test_non_text.count():,}")

        holdout_text, holdout_non_text = separate_text_and_non_text_features(holdout_data, spark)
        logger.info(f"  Holdout - Text: {holdout_text.count():,}, Non-text: {holdout_non_text.count():,}")

        stage_times[7] = datetime.now() - stage_start
        log_stage(7, TOTAL_STAGES, "Separate Text and Non-Text Features", "DONE")

        # STAGE 8: Save all datasets to Parquet
        log_stage(8, TOTAL_STAGES, "Save All Datasets to Parquet")
        stage_start = datetime.now()
        output_dir = str(PathConfig.get_etl_output_dir())
        logger.info(f"  Output directory: {output_dir}")

        save_to_parquet(train_text, output_dir, "train_text", spark)
        save_to_parquet(train_non_text, output_dir, "train_non_text", spark)
        save_to_parquet(test_text, output_dir, "test_text", spark)
        save_to_parquet(test_non_text, output_dir, "test_non_text", spark)
        save_to_parquet(holdout_text, output_dir, "holdout_text", spark)
        save_to_parquet(holdout_non_text, output_dir, "holdout_non_text", spark)

        stage_times[8] = datetime.now() - stage_start
        log_stage(8, TOTAL_STAGES, "Save All Datasets to Parquet", "DONE")

        # Cleanup checkpoint files (only exist for full runs)
        import shutil
        for checkpoint_name in ["_checkpoint_joined", "_checkpoint_discounted"]:
            checkpoint_path = PathConfig.get_etl_output_dir() / checkpoint_name
            if checkpoint_path.exists():
                logger.info(f"  Cleaning up checkpoint: {checkpoint_path}")
                shutil.rmtree(checkpoint_path)

        # Success summary
        total_elapsed = datetime.now() - start_time
        logger.info("")
        logger.info("=" * 80)
        logger.info("  ETL PIPELINE COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"  Finished at:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Total time:   {total_elapsed}")
        logger.info("")
        logger.info("  Stage Timings:")
        for stage_num, stage_time in stage_times.items():
            pct = (stage_time.total_seconds() / total_elapsed.total_seconds()) * 100
            logger.info(f"    Stage {stage_num}: {stage_time} ({pct:.1f}%)")
        logger.info("")
        logger.info("  Output files:")
        for f in ["train_text", "train_non_text", "test_text", "test_non_text", "holdout_text", "holdout_non_text"]:
            logger.info(f"    {output_dir}/{f}.parquet")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"ETL Pipeline failed: {e}", exc_info=True)
        raise

    finally:
        if "spark" in globals():
            stop_spark_session(spark)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Yelp Review ETL Pipeline - Stage 1: Data Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test run with 1000 samples
  python src/1_ETL_Data_Preparation.py --sample 1000

  # Full dataset run (overnight)
  python src/1_ETL_Data_Preparation.py --full

  # Default (10000 samples for development)
  python src/1_ETL_Data_Preparation.py
        """
    )
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=10000,
        help="Number of reviews to process (default: 10000)"
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Process full dataset (~8.6M reviews). Overrides --sample."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample_size = None if args.full else args.sample
    main(sample_size=sample_size)
