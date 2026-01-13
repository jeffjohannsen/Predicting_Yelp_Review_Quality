"""
Configuration management for Yelp Review Quality Prediction project.

This module provides centralized configuration for:
- Path management (all file paths in one place)
- Spark settings
- Model hyperparameters

Usage:
    from config import PathConfig, SparkConfig, ModelConfig

    train_path = PathConfig.get_train_csv()
    spark_master = SparkConfig.SPARK_MASTER
"""

import os
import socket
from pathlib import Path


class Config:
    """Base configuration class."""

    # Project root directory
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()

    @classmethod
    def detect_environment(cls):
        """Auto-detect the runtime environment."""
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

        if env == "ec2":
            return Path("/home/ubuntu")
        elif env == "jupyter":
            return Path("/home/jovyan/data")
        else:  # local
            return cls.PROJECT_ROOT / "data"

    @classmethod
    def get_models_dir(cls):
        """Get the models directory based on environment."""
        env = cls.detect_environment()

        if env == "ec2":
            return Path("/home/ubuntu/models")
        elif env == "jupyter":
            return Path("/home/jovyan/models")
        else:  # local
            return cls.PROJECT_ROOT / "models"


class PathConfig(Config):
    """Path configuration for data and models."""

    # === Source Data ===
    @classmethod
    def get_parquet_2021_dir(cls):
        """Get path to Parquet-converted 2021 dataset (6.7 GB)."""
        return cls.PROJECT_ROOT / "data" / "parquet_2021"

    @classmethod
    def get_parquet_business(cls):
        return cls.get_parquet_2021_dir() / "business.parquet"

    @classmethod
    def get_parquet_checkin(cls):
        return cls.get_parquet_2021_dir() / "checkin.parquet"

    @classmethod
    def get_parquet_review(cls):
        return cls.get_parquet_2021_dir() / "review.parquet"

    @classmethod
    def get_parquet_tip(cls):
        return cls.get_parquet_2021_dir() / "tip.parquet"

    @classmethod
    def get_parquet_user(cls):
        return cls.get_parquet_2021_dir() / "user.parquet"

    # === Pipeline Output Directories ===
    @classmethod
    def get_processed_dir(cls):
        """Get path to processed intermediate data."""
        return cls.PROJECT_ROOT / "data" / "processed"

    @classmethod
    def get_etl_output_dir(cls):
        """Stage 1 ETL output."""
        return cls.get_processed_dir() / "01_etl_output"

    @classmethod
    def get_nlp_output_dir(cls):
        """Stage 2 NLP output base directory."""
        return cls.get_processed_dir() / "02_nlp_output"

    @classmethod
    def get_nlp_basic_dir(cls):
        """Stage 2.1 - Basic text features."""
        return cls.get_nlp_output_dir() / "2.1_basic_text"

    @classmethod
    def get_nlp_spacy_dir(cls):
        """Stage 2.2 - spaCy linguistic features."""
        return cls.get_nlp_output_dir() / "2.2_spacy"

    @classmethod
    def get_nlp_tfidf_dir(cls):
        """Stage 2.3 - TF-IDF and ML predictions."""
        return cls.get_nlp_output_dir() / "2.3_tfidf"

    @classmethod
    def get_nlp_embeddings_dir(cls):
        """Stage 2.4 - FastText embeddings."""
        return cls.get_nlp_output_dir() / "2.4_embeddings"

    @classmethod
    def get_nlp_lda_dir(cls):
        """Stage 2.5 - LDA topic modeling."""
        return cls.get_nlp_output_dir() / "2.5_lda"

    @classmethod
    def get_model_ready_dir(cls):
        """Stage 3 output - Model-ready CSV files."""
        return cls.PROJECT_ROOT / "data" / "model_ready"

    @classmethod
    def get_final_predict_dir(cls):
        """Stage 7 output - Final predictions."""
        return cls.PROJECT_ROOT / "data" / "final_predict"

    # === Model Directories ===
    @classmethod
    def get_base_models_dir(cls):
        """Stage 4 - Base model comparison."""
        return cls.get_models_dir() / "base_models"

    @classmethod
    def get_final_models_dir(cls):
        """Stages 5-6 - Final trained models."""
        return cls.get_models_dir() / "final_models"

    @classmethod
    def get_nlp_models_dir(cls):
        """Stage 2 - NLP models (TF-IDF, FastText, LDA)."""
        return cls.get_models_dir() / "nlp"

    # === Convenience Methods ===
    @classmethod
    def get_train_csv(cls):
        return cls.get_model_ready_dir() / "train.csv"

    @classmethod
    def get_test_csv(cls):
        return cls.get_model_ready_dir() / "test.csv"

    @classmethod
    def get_holdout_csv(cls):
        return cls.get_model_ready_dir() / "holdout.csv"


class SparkConfig(Config):
    """Spark configuration."""

    # Use all but 1 CPU core by default
    SPARK_MASTER = os.getenv("SPARK_MASTER", "local[7]")


class ModelConfig:
    """Model training configuration and hyperparameters."""

    # Feature data types (for memory optimization when loading CSVs)
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
        "sentence_cnt": "int16",
        "avg_word_len": "float32",
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
    }


# Print config on import (for debugging)
if __name__ == "__main__":
    print("=== Yelp Review Quality Prediction - Configuration ===")
    print(f"Project Root: {Config.PROJECT_ROOT}")
    print(f"Environment:  {Config.detect_environment()}")
    print(f"Data Dir:     {PathConfig.get_data_dir()}")
    print(f"Models Dir:   {PathConfig.get_models_dir()}")
    print(f"Train CSV:    {PathConfig.get_train_csv()}")
    print(f"Parquet Dir:  {PathConfig.get_parquet_2021_dir()}")
