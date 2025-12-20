"""
Configuration management for Yelp Review Quality Prediction project.

This module provides centralized configuration for:
- Environment detection (local, EC2, Jupyter)
- Path management
- Database credentials
- Model hyperparameters
"""

import os
import socket
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Base configuration class."""

    # Project root directory
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()

    # Environment detection
    ENVIRONMENT = os.getenv("ENVIRONMENT", "auto")

    @classmethod
    def detect_environment(cls):
        """Auto-detect the runtime environment."""
        if cls.ENVIRONMENT != "auto":
            return cls.ENVIRONMENT

        hostname = socket.gethostname()

        # Check for EC2
        if hostname.startswith("ip-") or "ec2" in hostname.lower():
            return "ec2"

        # Check for Jupyter
        if "jovyan" in os.path.expanduser("~"):
            return "jupyter"

        # Default to local
        return "local"

    @classmethod
    def get_data_dir(cls):
        """Get the data directory based on environment."""
        env = cls.detect_environment()

        # Check for explicit path in .env
        if os.getenv("DATA_DIR"):
            return Path(os.getenv("DATA_DIR"))

        # Environment-specific paths
        if env == "ec2":
            return Path("/home/ubuntu")
        elif env == "jupyter":
            return Path("/home/jovyan/data")
        else:  # local
            return cls.PROJECT_ROOT / "data" / "full_data"

    @classmethod
    def get_models_dir(cls):
        """Get the models directory based on environment."""
        env = cls.detect_environment()

        # Check for explicit path in .env
        if os.getenv("MODELS_DIR"):
            return Path(os.getenv("MODELS_DIR"))

        # Environment-specific paths
        if env == "ec2":
            return Path("/home/ubuntu/models")
        elif env == "jupyter":
            return Path("/home/jovyan/models")
        else:  # local
            return cls.PROJECT_ROOT / "models"


class DatabaseConfig(Config):
    """Database configuration (legacy - AWS RDS no longer available)."""

    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_NAME = os.getenv("DB_NAME", "yelp_db")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")

    @classmethod
    def get_connection_string(cls):
        """Get PostgreSQL connection string."""
        return (
            f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}"
            f"@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
        )

    @classmethod
    def get_jdbc_url(cls):
        """Get JDBC URL for Spark."""
        return f"jdbc:postgresql://{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"


class PathConfig(Config):
    """Path configuration for data and models."""

    @classmethod
    def get_original_json_dir(cls):
        """Get path to original JSON files (2021 dataset)."""
        return cls.PROJECT_ROOT / "data" / "original_json_2021"

    @classmethod
    def get_parquet_2021_dir(cls):
        """Get path to Parquet-converted 2021 dataset."""
        return cls.PROJECT_ROOT / "data" / "parquet_2021"

    @classmethod
    def get_processed_dir(cls):
        """Get path to processed intermediate data (Parquet)."""
        return cls.PROJECT_ROOT / "data" / "processed"

    @classmethod
    def get_model_ready_dir(cls):
        """Get path to model-ready CSV files."""
        return cls.PROJECT_ROOT / "data" / "model_ready"

    @classmethod
    def get_final_predict_dir(cls):
        """Get path to final prediction output."""
        return cls.PROJECT_ROOT / "data" / "final_predict"

    @classmethod
    def get_base_models_dir(cls):
        """Get path to base models."""
        return cls.get_models_dir() / "base_models"

    @classmethod
    def get_final_models_dir(cls):
        """Get path to final models."""
        return cls.get_models_dir() / "final_models"

    @classmethod
    def get_nlp_models_dir(cls):
        """Get path to NLP models."""
        return cls.get_models_dir() / "nlp"

    @classmethod
    def get_train_csv(cls):
        """Get path to train.csv."""
        return cls.get_model_ready_dir() / "train.csv"

    @classmethod
    def get_test_csv(cls):
        """Get path to test.csv."""
        return cls.get_model_ready_dir() / "test.csv"

    # Parquet dataset paths
    @classmethod
    def get_parquet_business(cls):
        """Get path to business.parquet."""
        return cls.get_parquet_2021_dir() / "business.parquet"

    @classmethod
    def get_parquet_checkin(cls):
        """Get path to checkin.parquet."""
        return cls.get_parquet_2021_dir() / "checkin.parquet"

    @classmethod
    def get_parquet_review(cls):
        """Get path to review.parquet."""
        return cls.get_parquet_2021_dir() / "review.parquet"

    @classmethod
    def get_parquet_tip(cls):
        """Get path to tip.parquet."""
        return cls.get_parquet_2021_dir() / "tip.parquet"

    @classmethod
    def get_parquet_user(cls):
        """Get path to user.parquet."""
        return cls.get_parquet_2021_dir() / "user.parquet"


class SparkConfig(Config):
    """Spark configuration."""

    # Spark master
    SPARK_MASTER = os.getenv("SPARK_MASTER", "local[7]")

    # PostgreSQL JDBC driver path
    @classmethod
    def get_postgres_jar_path(cls):
        """Get path to PostgreSQL JDBC driver."""
        jar_locations = [
            cls.PROJECT_ROOT / "postgresql-42.2.20.jar",
            cls.PROJECT_ROOT / "lib" / "postgresql-42.2.20.jar",
        ]
        for jar in jar_locations:
            if jar.exists():
                return str(jar)
        return str(cls.PROJECT_ROOT / "postgresql-42.2.20.jar")


class ModelConfig:
    """Model training configuration and hyperparameters."""

    # Data loading defaults
    DEFAULT_TRAIN_RECORDS = 1_000_000
    DEFAULT_TEST_RECORDS = 1_000_000

    # Feature data types (for memory optimization)
    FEATURE_DTYPES = {
        "target_reg": "int16",
        "review_stars": "int16",
        "NB_prob": "float32",
        "svm_pred": "float32",
        "ft_prob": "float32",
        "lda_t1": "float32",
        "lda_t2": "float32",
        "lda_t3": "float32",
        "lda_t4": "float32",
        "lda_t5": "float32",
        "grade_level": "float32",
        "polarity": "float32",
        "subjectivity": "float32",
        "word_cnt": "int16",
        "character_cnt": "int16",
        "num_cnt": "int16",
        "uppercase_cnt": "int16",
        "#@_cnt": "int16",
        "sentence_cnt": "int16",
        "lexicon_cnt": "int16",
        "syllable_cnt": "int16",
        "avg_word_len": "float32",
        "token_cnt": "int16",
        "stopword_cnt": "int16",
        "stopword_pct": "float32",
        "ent_cnt": "int16",
        "ent_pct": "float32",
    }

    # Logistic Regression hyperparameters
    LOGREG_PARAMS = {
        "C": [0.5, 1.0, 2.0],
        "solver": ["lbfgs", "newton-cg", "saga"],
        "penalty": ["l2"],
        "max_iter": [50, 100, 5000],
        "class_weight": [None, "balanced"],
    }

    # Random Forest hyperparameters
    RF_PARAMS = {
        "n_estimators": [5, 10],
        "max_depth": [10, 100, None],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 5],
        "max_features": ["sqrt", 10, None],
        "max_leaf_nodes": [10, 100, None],
        "criterion": ["gini", "entropy"],
    }

    # Model naming
    MODEL_NAME_POSTFIX = "_sklearn_model"


# Convenience functions
def get_db_connection_string():
    """Get database connection string."""
    return DatabaseConfig.get_connection_string()


def get_train_csv_path():
    """Get path to train.csv."""
    return PathConfig.get_train_csv()


def get_test_csv_path():
    """Get path to test.csv."""
    return PathConfig.get_test_csv()


# Print config on import (for debugging)
if __name__ == "__main__":
    print("=== Yelp Review Quality Prediction - Configuration ===")
    print(f"Project Root: {Config.PROJECT_ROOT}")
    print(f"Data Directory: {PathConfig.get_data_dir()}")
    print(f"Models Directory: {PathConfig.get_models_dir()}")
    print(f"Train CSV: {PathConfig.get_train_csv()}")
    print(f"Test CSV: {PathConfig.get_test_csv()}")
