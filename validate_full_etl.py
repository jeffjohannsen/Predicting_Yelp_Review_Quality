#!/usr/bin/env python
"""
Validation Script for Full ETL Output
Run this in the morning to verify the overnight ETL completed successfully.

Usage:
    python validate_full_etl.py           # Validate full dataset (expects ~8.6M rows)
    python validate_full_etl.py --sample  # Validate sample run (any size)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import PathConfig


def print_header(title):
    """Print a formatted header."""
    print("")
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title):
    """Print a section header."""
    print("")
    print(f"  {title}")
    print("  " + "-" * (len(title) + 2))


def validate_etl(expect_full=True):
    """
    Validate ETL output files.

    Args:
        expect_full: If True, expects ~8.6M rows (full dataset).
                    If False, accepts any sample size.

    Returns:
        True if validation passes, False otherwise.
    """
    from datetime import datetime

    print_header("FULL ETL VALIDATION REPORT")
    print(f"  Validation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Expected size:   {'Full dataset (~8.6M rows)' if expect_full else 'Sample (any size)'}")

    base_path = PathConfig.get_etl_output_dir()

    # Check 1: Output directory exists
    print_section("1. OUTPUT DIRECTORY CHECK")
    if not base_path.exists():
        print(f"  ✗ FAILED: Output directory does not exist!")
        print(f"    Expected: {base_path}")
        return False
    print(f"  ✓ Output directory exists: {base_path}")

    # Check 2: All expected files exist
    print_section("2. OUTPUT FILES CHECK")
    expected_files = [
        "train_text.parquet",
        "train_non_text.parquet",
        "test_text.parquet",
        "test_non_text.parquet",
        "holdout_text.parquet",
        "holdout_non_text.parquet",
    ]

    missing_files = []
    file_sizes = {}
    for filename in expected_files:
        file_path = base_path / filename
        if file_path.exists():
            # Parquet directories contain multiple part files
            if file_path.is_dir():
                size_bytes = sum(
                    f.stat().st_size for f in file_path.glob("**/*") if f.is_file()
                )
            else:
                size_bytes = file_path.stat().st_size
            size_mb = size_bytes / 1024 / 1024
            file_sizes[filename] = size_mb
            print(f"  ✓ {filename:30} {size_mb:8.1f} MB")
        else:
            print(f"  ✗ {filename:30} MISSING")
            missing_files.append(filename)

    if missing_files:
        print(f"\n  ✗ FAILED: {len(missing_files)} files missing: {missing_files}")
        return False

    # Check 3: Data sizes and split ratios
    print_section("3. DATA SIZES AND SPLIT RATIOS")
    try:
        train_text = pd.read_parquet(base_path / "train_text.parquet")
        train_non_text = pd.read_parquet(base_path / "train_non_text.parquet")
        test_text = pd.read_parquet(base_path / "test_text.parquet")
        test_non_text = pd.read_parquet(base_path / "test_non_text.parquet")
        holdout_text = pd.read_parquet(base_path / "holdout_text.parquet")
        holdout_non_text = pd.read_parquet(base_path / "holdout_non_text.parquet")

        total = len(train_text) + len(test_text) + len(holdout_text)
        train_pct = len(train_text) / total * 100
        test_pct = len(test_text) / total * 100
        holdout_pct = len(holdout_text) / total * 100

        print(f"  Train:   {len(train_text):>10,} rows ({train_pct:5.1f}%)")
        print(f"  Test:    {len(test_text):>10,} rows ({test_pct:5.1f}%)")
        print(f"  Holdout: {len(holdout_text):>10,} rows ({holdout_pct:5.1f}%)")
        print(f"  Total:   {total:>10,} rows")

        # Verify split ratios are approximately 70/15/15
        if not (65 < train_pct < 75 and 12 < test_pct < 18 and 12 < holdout_pct < 18):
            print(f"\n  ⚠️  WARNING: Split ratios deviate from expected 70/15/15")

        # Verify total count for full dataset
        if expect_full:
            if total < 8_000_000:
                print(f"\n  ✗ FAILED: Expected ~8.6M rows, got {total:,}")
                print(f"    This might be a sample run. Use --sample flag if intentional.")
                return False
            print(f"\n  ✓ Row count looks good (~8.6M expected)")
        else:
            print(f"\n  ✓ Sample validation - row count: {total:,}")

    except Exception as e:
        print(f"  ✗ FAILED: Could not load data: {e}")
        return False

    # Check 4: Temporal split validation
    print_section("4. TEMPORAL SPLIT VALIDATION")
    try:
        train_dates = pd.to_datetime(train_non_text["review_date"])
        test_dates = pd.to_datetime(test_non_text["review_date"])
        holdout_dates = pd.to_datetime(holdout_non_text["review_date"])

        print(f"  Train:   {train_dates.min().date()} to {train_dates.max().date()}")
        print(f"  Test:    {test_dates.min().date()} to {test_dates.max().date()}")
        print(f"  Holdout: {holdout_dates.min().date()} to {holdout_dates.max().date()}")

        # Check for temporal leakage
        if (
            train_dates.max() < test_dates.min()
            and test_dates.max() < holdout_dates.min()
        ):
            print("\n  ✓ No temporal leakage - Perfect chronological split!")
        else:
            print("\n  ⚠️  WARNING: Dates overlap - possible temporal leakage")
            print(f"    Train max: {train_dates.max()}")
            print(f"    Test min:  {test_dates.min()}")
            # Don't fail on this - could be intentional random split

    except Exception as e:
        print(f"  ✗ FAILED: Could not validate temporal split: {e}")
        return False

    # Check 5: Data quality checks
    print_section("5. DATA QUALITY CHECKS")

    checks = []

    # Critical columns
    checks.append(("No nulls in review_id", train_text["review_id"].isnull().sum() == 0))
    checks.append(
        ("No nulls in review_text", train_text["review_text"].isnull().sum() == 0)
    )
    checks.append(("No duplicates in train", not train_text["review_id"].duplicated().any()))

    # Target variables
    checks.append(
        ("T1 (raw UFC) non-negative", (train_text["T1_REG_review_total_ufc"] >= 0).all())
    )
    checks.append(
        ("T4 (time-discounted) non-negative", (train_text["T4_REG_ufc_TD"] >= 0).all())
    )
    checks.append(
        (
            "T4 <= T1 (discounting reduces)",
            (train_text["T4_REG_ufc_TD"] <= train_text["T1_REG_review_total_ufc"]).all(),
        )
    )

    # Text quality
    checks.append(
        ("No empty review texts", (train_text["review_text"].str.len() > 0).all())
    )

    # Column consistency
    checks.append(
        (
            "Text/Non-text row counts match",
            len(train_text) == len(train_non_text),
        )
    )

    all_passed = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False

    if not all_passed:
        failed_checks = [name for name, passed in checks if not passed]
        print(f"\n  ✗ FAILED {len(failed_checks)} checks: {failed_checks}")
        return False

    # Check 6: Feature summary
    print_section("6. FEATURE SUMMARY")
    print(f"  Text columns:     {len(train_text.columns)}")
    print(f"  Non-text columns: {len(train_non_text.columns)}")

    target_cols = [col for col in train_text.columns if col.startswith("T")]
    print(f"  Target variables: {target_cols}")

    # Show sample of time-discounted features
    td_cols = [col for col in train_non_text.columns if col.endswith("_TD")]
    print(f"  Time-discounted features: {len(td_cols)}")
    for col in td_cols[:5]:
        print(f"    - {col}")
    if len(td_cols) > 5:
        print(f"    ... and {len(td_cols) - 5} more")

    # Final summary
    print_header("VALIDATION RESULT")
    print("  ✓ ALL CHECKS PASSED - ETL COMPLETED SUCCESSFULLY!")
    print("")
    print("  Next steps:")
    print("    1. Review the log file for any warnings")
    print("    2. Proceed to Stage 2 NLP feature engineering")
    print("       - src/2.1_NLP_Basic_Text.py")
    print("       - src/2.2_NLP_Spacy_Linguistic.py")
    print("       - etc.")
    print("")
    print("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate ETL output files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_full_etl.py           # Expect full dataset (~8.6M rows)
  python validate_full_etl.py --sample  # Accept any sample size
        """,
    )
    parser.add_argument(
        "--sample",
        "-s",
        action="store_true",
        help="Validate sample run (accept any row count)",
    )
    args = parser.parse_args()

    success = validate_etl(expect_full=not args.sample)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
