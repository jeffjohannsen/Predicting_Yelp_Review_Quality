# Data Storage Strategy Decision

## Context

**Problem**: All current scripts save to PostgreSQL on AWS RDS, which no longer exists.

**Data Scale**:
- Original JSON: 12 GB (8.6M reviews)
- Need to process and store intermediate results
- Need to handle distributed processing with Spark

**Current State**:
- Scripts use PySpark for distributed processing
- Save results to PostgreSQL tables
- Load from PostgreSQL for next stage

---

## Storage Options Analysis

### Option 1: **Parquet Files (RECOMMENDED)**

**Pros:**
- ✅ Column-oriented format (excellent for analytics)
- ✅ Built-in compression (~5-10x smaller than CSV)
- ✅ Native Spark support (`.write.parquet()`, `.read.parquet()`)
- ✅ Preserves data types (no type inference issues)
- ✅ Fast read/write with Spark
- ✅ Can partition by columns (e.g., by date ranges)
- ✅ Industry standard for big data pipelines
- ✅ Works seamlessly with pandas via PyArrow

**Cons:**
- ❌ Not human-readable (binary format)
- ❌ Requires PyArrow library
- ❌ Slightly more setup than CSV

**Use Case:** Intermediate pipeline data (NLP features, transformed datasets)

**Example Structure:**
```
data/processed/
  01_etl_output/
    train.parquet/       # Partitioned parquet
    test.parquet/
    holdout.parquet/
  02_nlp_basic/
    train_basic_features.parquet/
    test_basic_features.parquet/
  02_nlp_spacy/
    train_spacy_features.parquet/
  ...
  03_combined/
    train_all_features.parquet/
    test_all_features.parquet/
```

**Spark Code:**
```python
# Write
df.write.mode('overwrite').parquet(str(paths.processed / "01_etl_output" / "train.parquet"))

# Read
df = spark.read.parquet(str(paths.processed / "01_etl_output" / "train.parquet"))
```

---

### Option 2: **CSV Files**

**Pros:**
- ✅ Human-readable
- ✅ Universal format (Excel, R, Python, etc.)
- ✅ No special libraries required
- ✅ Easy debugging (can open and inspect)
- ✅ Spark supports CSV

**Cons:**
- ❌ Large file sizes (no compression)
- ❌ Slow to read/write (especially with Spark)
- ❌ Type inference issues (strings vs numbers)
- ❌ Header parsing can be slow
- ❌ No built-in partitioning
- ❌ Poor for wide datasets (200+ columns)

**Use Case:** Final model-ready datasets (train.csv, test.csv) for compatibility

**Example Structure:**
```
data/model_ready/
  train.csv              # Final combined dataset
  test.csv
  holdout.csv
```

---

### Option 3: **Local PostgreSQL Database**

**Pros:**
- ✅ Relational structure (joins, indexes)
- ✅ SQL query interface
- ✅ Transaction support
- ✅ Can replicate old approach

**Cons:**
- ❌ Requires PostgreSQL installation and management
- ❌ Additional complexity (database server, connections)
- ❌ Slower than Parquet for analytics
- ❌ Overkill for linear pipeline (no complex queries needed)
- ❌ Connection overhead for Spark
- ❌ Backup/restore complexity

**Use Case:** Only if you need relational queries (not needed for this pipeline)

---

### Option 4: **SQLite Database**

**Pros:**
- ✅ No server needed (file-based)
- ✅ SQL queries available
- ✅ Portable (single file)
- ✅ No configuration

**Cons:**
- ❌ Poor Spark integration (no native connector)
- ❌ Single-threaded writes (slow for big data)
- ❌ Not designed for analytics workloads
- ❌ Limited concurrent access

**Use Case:** Small reference tables only (not for main pipeline)

---

### Option 5: **Hybrid Approach (RECOMMENDED)**

Combine the best of multiple options:

**Parquet for Pipeline Stages:**
```
data/processed/
  01_etl_output/          # Parquet (Spark output)
  02_nlp_*/               # Parquet (intermediate NLP features)
  03_combined/            # Parquet (all features joined)
```

**CSV for Final Datasets:**
```
data/model_ready/
  train.csv               # Final dataset for ML
  test.csv
  holdout.csv
```

**Why Hybrid:**
- Parquet: Fast Spark processing, small files, type-safe
- CSV: Final datasets for compatibility with any ML tool
- Best of both worlds

---

## Recommendation: **Hybrid Parquet + CSV**

### Implementation Plan

**Stage 1-2 (ETL, NLP):** Save to Parquet
- Use Spark's native `.write.parquet()`
- Partition by logical chunks if needed
- Preserve data types
- Fast processing

**Stage 3 (Feature Combination):** Save to Parquet first
- Combine all features with Spark
- Save combined dataset as Parquet
- Then export to CSV for model training

**Stage 4+ (ML):** Use CSV
- Load from `data/model_ready/*.csv`
- Train models with pandas/sklearn
- Compatible with all ML libraries

### Directory Structure

```
data/
  original_json_2021/           # Raw Yelp JSON (12 GB)
  original_json_2024/           # Future use
  processed/                    # NEW - Parquet intermediate data
    01_etl_output/
      metadata.json             # Record counts, creation date
      train.parquet/
      test.parquet/
      holdout.parquet/
    02_nlp_basic/
      train.parquet/
      test.parquet/
    02_nlp_spacy/
      train.parquet/
      test.parquet/
    02_nlp_tfidf/
      train_nb_predictions.parquet/
      train_svm_predictions.parquet/
    02_nlp_embeddings/
      train_fasttext.parquet/
    02_nlp_lda/
      train_topics.parquet/
    03_combined/
      train_all_features.parquet/
      test_all_features.parquet/
  model_ready/                  # NEW - CSV final datasets
    train.csv
    test.csv
    holdout.csv
    feature_names.txt           # List of all features
    metadata.json               # Dataset info
  final_predict/                # Model predictions
    test_predictions.csv
    holdout_predictions.csv
```

### Code Pattern

**In each script:**

```python
from pathlib import Path
from config import PathConfig

paths = PathConfig()

# Read previous stage (Parquet)
train_df = spark.read.parquet(
    str(paths.processed / "01_etl_output" / "train.parquet")
)

# Process data
# ... NLP processing ...

# Write current stage (Parquet)
result_df.write.mode('overwrite').parquet(
    str(paths.processed / "02_nlp_basic" / "train.parquet")
)

# Optional: Export to CSV for inspection
result_df.coalesce(1).write.mode('overwrite').csv(
    str(paths.processed / "02_nlp_basic" / "train_sample.csv"),
    header=True
)
```

**For final stage (3_ETL_Combine):**

```python
# Combine all features (Parquet)
combined_df.write.mode('overwrite').parquet(
    str(paths.processed / "03_combined" / "train_all_features.parquet")
)

# Export to CSV for ML
combined_df.write.mode('overwrite').csv(
    str(paths.model_ready / "train.csv"),
    header=True
)
```

---

## Migration Steps

1. **Update `config.py`** - Add paths for processed/ and model_ready/
2. **Create directories** - Set up new data structure
3. **Update `1_ETL_Spark.py`** - Save to Parquet instead of PostgreSQL
4. **Update all `2.x` scripts** - Read from Parquet, save to Parquet
5. **Update `3_*.sql`** - Convert SQL to PySpark, save to Parquet + CSV
6. **Update all ML scripts** - Read from CSV in model_ready/
7. **Remove PostgreSQL imports** - Clean up all scripts
8. **Add metadata tracking** - JSON files with record counts, dates

---

## Requirements

### Python Packages Needed

```txt
# Already in requirements.txt (hopefully)
pyspark>=3.0.0
pandas>=1.0.0
pyarrow>=5.0.0          # For Parquet support in pandas
fastparquet             # Alternative Parquet engine (optional)

# For compression (optional but recommended)
snappy                  # Faster compression
```

### Disk Space Estimates

Based on 12 GB raw JSON:
- Parquet intermediate: ~3-5 GB total (all stages)
- CSV final: ~2-3 GB (compressed with gzip)
- Total new data: ~8 GB maximum

### Performance Benefits

- **Parquet reads**: 5-10x faster than CSV with Spark
- **File sizes**: 5-10x smaller than CSV
- **Type safety**: No parsing errors
- **Compression**: Built-in snappy/gzip

---

## Decision Required

**Do you want to:**

1. ✅ **Hybrid Parquet + CSV** (recommended)
   - Parquet for all intermediate pipeline stages
   - CSV for final model-ready datasets
   
2. **All Parquet** (most efficient)
   - Everything in Parquet
   - Convert to CSV only when needed for external tools
   
3. **All CSV** (simplest but slowest)
   - Everything in CSV
   - Easier to debug but much larger files

**My recommendation: Option 1 (Hybrid)** - gives you speed and compatibility.

Once decided, I'll update the config.py with the new paths and we can start refactoring the pipeline.

