"""
Spark Session Utilities for Yelp Review Quality Prediction pipeline.

This module provides utilities for creating and configuring Spark sessions
with appropriate settings for different environments (local development,
EC2, Jupyter notebooks).

Functions:
    create_spark_session: Create configured SparkSession for pipeline use
    get_optimal_cores: Determine optimal core count for local execution
"""

import logging
import multiprocessing
from typing import Optional

import pyspark as ps
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def get_optimal_cores() -> int:
    """
    Determine optimal number of cores for Spark local execution.

    Uses CPU count minus 1 to leave one core for OS and other processes.
    Minimum of 2 cores to ensure parallelism.

    Returns:
        Number of cores to use for Spark master local[N]

    Example:
        >>> cores = get_optimal_cores()
        >>> print(f"Using {cores} cores for Spark")
    """
    total_cores = multiprocessing.cpu_count()
    # Leave one core for system, min of 2
    optimal = max(2, total_cores - 1)
    return optimal


def create_spark_session(
    app_name: str = "Yelp_ETL",
    master: Optional[str] = None,
    memory_driver: str = "4g",
    memory_executor: str = "4g",
    enable_hive: bool = False,
    jdbc_jar_path: Optional[str] = None,
    log_level: str = "WARN",
    **extra_configs,
) -> SparkSession:
    """
    Create and configure a Spark session for the Yelp pipeline.

    Creates a SparkSession with sensible defaults for the Yelp review
    processing pipeline. Handles configuration for local development,
    EC2, and distributed processing.

    Args:
        app_name: Name for the Spark application (default: "Yelp_ETL")
        master: Spark master URL. If None, uses local[N] with optimal cores.
                Examples: "local[*]", "local[4]", "spark://host:7077"
        memory_driver: Driver memory allocation (default: "4g")
        memory_executor: Executor memory allocation (default: "4g")
        enable_hive: Enable Hive support (default: False)
        jdbc_jar_path: Path to PostgreSQL JDBC driver (if needed for legacy code)
        log_level: Spark logging level (default: "WARN")
                  Options: "ALL", "DEBUG", "ERROR", "FATAL", "INFO", "OFF", "TRACE", "WARN"
        **extra_configs: Additional Spark configuration options as key=value pairs

    Returns:
        Configured SparkSession instance

    Raises:
        Exception: If SparkSession creation fails

    Example:
        >>> # Basic usage
        >>> spark = create_spark_session()

        >>> # Explicit core count
        >>> spark = create_spark_session(master="local[8]")

        >>> # With extra configuration
        >>> spark = create_spark_session(
        ...     app_name="Yelp_NLP",
        ...     memory_driver="8g",
        ...     **{"spark.sql.shuffle.partitions": "200"}
        ... )

    Notes:
        - Default master is local[N] where N = CPU cores - 1
        - Session is created with .getOrCreate() to reuse existing sessions
        - Driver and executor memory should be adjusted based on dataset size
        - For full 8M review dataset, recommend 8g+ driver memory
    """
    try:
        # Determine master URL
        if master is None:
            cores = get_optimal_cores()
            master = f"local[{cores}]"
            logger.info(f"Auto-detected master: {master}")

        # Start building session
        builder = SparkSession.builder.appName(app_name).master(master)

        # Memory configuration
        builder = builder.config("spark.driver.memory", memory_driver)
        builder = builder.config("spark.executor.memory", memory_executor)

        # JDBC driver path (if provided for PostgreSQL legacy support)
        if jdbc_jar_path:
            builder = builder.config("spark.driver.extraClassPath", jdbc_jar_path)
            logger.info(f"JDBC driver configured: {jdbc_jar_path}")

        # Hive support (if needed)
        if enable_hive:
            builder = builder.enableHiveSupport()
            logger.info("Hive support enabled")

        # Apply extra configurations
        for key, value in extra_configs.items():
            builder = builder.config(key, value)
            logger.debug(f"Extra config: {key}={value}")

        # Create or get existing session
        spark = builder.getOrCreate()

        # Set log level
        spark.sparkContext.setLogLevel(log_level)

        logger.info(
            f"SparkSession created: {app_name} | Master: {master} | "
            f"Driver: {memory_driver} | Executor: {memory_executor}"
        )

        return spark

    except Exception as e:
        logger.error(f"Failed to create SparkSession: {e}")
        raise


def stop_spark_session(spark: SparkSession) -> None:
    """
    Safely stop a Spark session.

    Args:
        spark: SparkSession to stop

    Example:
        >>> spark = create_spark_session()
        >>> # ... do work ...
        >>> stop_spark_session(spark)
    """
    if spark:
        try:
            spark.stop()
            logger.info("SparkSession stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping SparkSession: {e}")


def get_spark_config_summary(spark: SparkSession) -> dict:
    """
    Get summary of current Spark configuration.

    Useful for debugging and verifying Spark settings.

    Args:
        spark: SparkSession to inspect

    Returns:
        Dictionary with key configuration values

    Example:
        >>> spark = create_spark_session()
        >>> config = get_spark_config_summary(spark)
        >>> print(f"Using {config['cores']} cores")
    """
    conf = spark.sparkContext.getConf()

    return {
        "app_name": conf.get("spark.app.name"),
        "master": conf.get("spark.master"),
        "driver_memory": conf.get("spark.driver.memory"),
        "executor_memory": conf.get("spark.executor.memory"),
        "cores": conf.get("spark.executor.cores", "auto"),
        "log_level": spark.sparkContext.getConf().get("spark.logLevel", "INFO"),
    }


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    print("Creating Spark session...")
    spark = create_spark_session(
        app_name="Test_Session",
        memory_driver="2g",
        memory_executor="2g",
        log_level="ERROR",
    )

    print("\nSpark Configuration:")
    config = get_spark_config_summary(spark)
    for key, value in config.items():
        print(f"  {key}: {value}")

    print(f"\nSpark Version: {spark.version}")
    print(f"Python Version: {spark.sparkContext.pythonVer}")

    # Test basic functionality
    print("\nTesting basic Spark functionality...")
    test_data = [(1, "a"), (2, "b"), (3, "c")]
    df = spark.createDataFrame(test_data, ["id", "value"])
    print(f"Created test DataFrame with {df.count()} rows")
    df.show()

    print("\nStopping Spark session...")
    stop_spark_session(spark)
    print("Done!")
