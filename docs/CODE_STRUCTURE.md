# Code Structure Overview

**Generated**: December 19, 2025  
**Purpose**: Understanding the project's codebase organization

---

## Executive Summary

The project has **3 distinct code versions** with significant overlap:

1. **Current/Active Scripts** (`src/`) - 13 production scripts (4,813 lines)
2. **Version 1** (`src/version_1/` + `notebooks/version_1/`) - Old pipeline approach
3. **Version 0** (`src/version_0/`) - Original MongoDB-based pipeline (830 lines)

Plus **19 Jupyter notebooks** in the main notebooks directory for exploration and development.

---

## Current Active Code (`src/`)

### Production Scripts (13 files, 4,813 lines)

#### 1. ETL Pipeline (1,373 lines)
| File | Lines | Purpose |
|------|-------|---------|
| `1_ETL_Spark.py` | 337 | Main ETL: JSON → PostgreSQL (Spark) |
| `3_ETL_Combine_Processed_Text_Data.sql` | 400 | SQL: Join all text features |
| `11.1_ETL_Add_Predictions.py` | 600 | Add model predictions to datasets |

**Flow**: Raw JSON → Spark processing → PostgreSQL → Feature tables → Combined dataset

#### 2. NLP Pipeline (1,331 lines)
| File | Lines | Purpose |
|------|-------|---------|
| `2.1_NLP_Basic_Text_Processing_Spark.py` | 204 | Basic text features (TextBlob, readability) |
| `2.2_NLP_Spacy_POS_ENT_DEP.py` | 236 | spaCy linguistic analysis (POS, NER, dependencies) |
| `2.3.1_NLP_Spark_Tf-Idf_Models.py` | 252 | TF-IDF + Naive Bayes/SVM classification |
| `2.4.1_NLP_Spark_Text_Embeddings.py` | 356 | Word embeddings (Spark NLP) |
| `2.4.2_NLP_Fasttext.py` | 109 | FastText embeddings |
| `2.5_NLP_Topic_Modeling_LDA.py` | 174 | LDA topic modeling (Gensim) |

**Flow**: PostgreSQL text → Feature extraction → Feature tables → Combined features

#### 3. Machine Learning Pipeline (2,109 lines)
| File | Lines | Purpose |
|------|-------|---------|
| `4.1_ML_Base_Models.py` | 336 | Baseline models (LogReg, Tree, Forest, XGBoost) |
| `5_PCA_Dimensionality_Reduction.py` | 290 | PCA for feature reduction |
| `6.1_ML_Logistic_Regression.py` | 637 | Tuned logistic regression with CV |
| `10.1_ML_Linear_Regression.py` | 626 | Linear regression for continuous targets |
| `11.1_ETL_Add_Predictions.py` | 600 | (counted in ETL, adds predictions to data) |

**Flow**: Model-ready CSV → Train models → Evaluate → Tune → Save predictions

#### 4. Configuration
| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | 256 | Centralized configuration (NEW - added during cleanup) |

**Purpose**: Path management, model configs, database settings (replaces hardcoded paths)

---

## Jupyter Notebooks (19 files)

### Active Notebooks (Main Directory)

#### ETL & Data Prep (4 notebooks)
| Notebook | Size | Purpose |
|----------|------|---------|
| `1_ETL_Spark.ipynb` | 74K | Interactive ETL development |
| `3.1_ETL_Combined_Data_to_CSV.ipynb` | 47K | Export processed data to CSV |
| `3.2_EDA_Processed_Text_Data.ipynb` | 2.9MB | EDA on processed text features (LARGE) |
| `non_text_eda.ipynb` | 1.6MB | EDA on non-text features (LARGE) |

#### NLP Series (9 notebooks)
| Notebook | Size | Purpose |
|----------|------|---------|
| `2.0_NLP_Wordclouds.ipynb` | 551K | Text visualization (word clouds) |
| `2.1_NLP_Spark_Text_Basic.ipynb` | 19K | Basic text features |
| `2.2_NLP_Spacy_POS_ENT_DEP.ipynb` | 7.9K | spaCy linguistic analysis |
| `2.3_NLP_Tf-Idf_Models.ipynb` | 18K | TF-IDF + classification (non-Spark) |
| `2.3.1_NLP_Spark_Tf-Idf_Models.ipynb` | 18K | TF-IDF + classification (Spark version) |
| `2.4_NLP_Embeddings.ipynb` | 49K | Word embeddings (non-Spark) |
| `2.4.1_NLP_Spark_Embeddings.ipynb` | 29K | Word embeddings (Spark version) |
| `2.5_NLP_Topic_Modeling_LDA.ipynb` | 20K | LDA topic modeling |

**Pattern**: Notebooks often have both non-Spark and Spark versions (e.g., 2.3 vs 2.3.1)

#### Modeling & ML (6 notebooks)
| Notebook | Size | Purpose |
|----------|------|---------|
| `4.0_AutoML_PyCaret.ipynb` | 152K | AutoML experiments |
| `5.0_PCA_Dimensionality_Reduction.ipynb` | 118K | Dimensionality reduction |
| `5.1_Feature_Selection.ipynb` | 447K | Feature selection (classification) |
| `7.0_REG_Target_Adjustments.ipynb` | 608K | Target engineering for regression |
| `8.0_AutoML_PyCaret_Regression.ipynb` | 144K | AutoML for regression |
| `9.1_Feature_Selection_Regression.ipynb` | 544K | Feature selection (regression) |
| `12.1_Review_Ranking.ipynb` | 30K | Review ranking evaluation |

---

## Legacy Code (Version 1)

### Source Code (`src/version_1/`)

#### Main Scripts (4 files, 82,810 bytes)
| File | Lines | Purpose |
|------|-------|---------|
| `a1_Model_Pipeline.py` | ~1,000 | Main model training pipeline |
| `a2_NLP.py` | ~500 | NLP processing functions |
| `a3_Random_Forest_Metrics.py` | ~700 | Random Forest evaluation |
| `a4_Model_Setup.py` | ~650 | Model configuration and setup |

#### Helper Scripts (`python_scripts/` - 6 files)
| File | Purpose |
|------|---------|
| `business_checkins.py` | Business check-in feature engineering |
| `eda_prep.py` | EDA preparation utilities |
| `feature_engineering.py` | Feature creation functions |
| `json2sql.py` | JSON to SQL conversion |
| `migrate_db_2_aws.py` | Database migration to AWS |
| `user_connectivity.py` | User social network features |

#### SQL Scripts (`sql/`)
- `PostgreSQL_Command_Storage.sql` - 8,614 lines of SQL commands

### Notebooks (`notebooks/version_1/`)
| Notebook | Size | Purpose |
|----------|------|---------|
| `01_Exploratory_Data_Analysis.ipynb` | 8.3MB | **HUGE** - comprehensive EDA |
| `02_Data_Prep_Pipeline.ipynb` | 320KB | Data preparation pipeline |
| `eda_prep.ipynb` | 11KB | EDA utilities |

**Note**: The 8.3MB EDA notebook is likely the original comprehensive analysis

---

## Legacy Code (Version 0)

### Source Code (`src/version_0/` - 4 files, 830 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `y1_json2mongo.py` | 71 | Load JSON → MongoDB |
| `y1_mongo2sql.py` | 264 | Migrate MongoDB → PostgreSQL |
| `y1_model_prep.py` | 227 | Prepare data for modeling |
| `y1_pipeline.py` | 268 | Data cleaning pipeline (class-based) |

**Architecture**: JSON → MongoDB → PostgreSQL → Model-ready data

**Status**: Obsolete (MongoDB dependency removed)

---

## Code Evolution Timeline

### Version 0 (Original - MongoDB-based)
```
JSON files
    ↓
MongoDB (NoSQL storage)
    ↓
PostgreSQL (via migration)
    ↓
Model-ready data
```
- **Approach**: Two-database system (MongoDB + PostgreSQL)
- **Status**: Abandoned (MongoDB no longer used)
- **Lines of Code**: 830

### Version 1 (Intermediate - AWS-based)
```
JSON files
    ↓
PostgreSQL (AWS RDS)
    ↓
Feature engineering scripts
    ↓
Model-ready data
```
- **Approach**: Direct to PostgreSQL, helper scripts for features
- **Infrastructure**: AWS RDS (cloud database)
- **Status**: Semi-obsolete (AWS account closed)
- **Code Split**: Main scripts + helpers in subdirectories
- **Lines of Code**: ~3,000+ (estimated)

### Current (Production - Spark-based)
```
JSON files
    ↓
Spark ETL → PostgreSQL
    ↓
Spark NLP processing
    ↓
SQL feature combination
    ↓
Model-ready CSV
    ↓
sklearn/XGBoost models
```
- **Approach**: Spark for distributed processing, PostgreSQL for intermediate storage
- **Infrastructure**: Local + AWS RDS (no longer available)
- **Status**: **ACTIVE** but needs refactoring
- **Lines of Code**: 4,813
- **Notebooks**: 19 (for development/exploration)

---

## Code Relationships

### Notebook → Script Pairs

Many notebooks have corresponding `.py` scripts:

| Notebook | Script | Relationship |
|----------|--------|--------------|
| `1_ETL_Spark.ipynb` | `1_ETL_Spark.py` | Development → Production |
| `2.1_NLP_Spark_Text_Basic.ipynb` | `2.1_NLP_Basic_Text_Processing_Spark.py` | Development → Production |
| `2.2_NLP_Spacy_POS_ENT_DEP.ipynb` | `2.2_NLP_Spacy_POS_ENT_DEP.py` | Development → Production |
| `2.3.1_NLP_Spark_Tf-Idf_Models.ipynb` | `2.3.1_NLP_Spark_Tf-Idf_Models.py` | Development → Production |
| `2.4.1_NLP_Spark_Embeddings.ipynb` | `2.4.1_NLP_Spark_Text_Embeddings.py` | Development → Production |
| `2.5_NLP_Topic_Modeling_LDA.ipynb` | `2.5_NLP_Topic_Modeling_LDA.py` | Development → Production |
| `5.0_PCA_Dimensionality_Reduction.ipynb` | `5_PCA_Dimensionality_Reduction.py` | Development → Production |

**Pattern**: 
- Notebooks used for exploration, development, visualization
- `.py` scripts are productionized versions (often cleaner, less output)
- Some notebooks don't have script equivalents (AutoML, EDA-only)

### Duplicate Notebooks (Spark vs Non-Spark)

| Non-Spark | Spark | Status |
|-----------|-------|--------|
| `2.3_NLP_Tf-Idf_Models.ipynb` | `2.3.1_NLP_Spark_Tf-Idf_Models.ipynb` | Spark version is production |
| `2.4_NLP_Embeddings.ipynb` | `2.4.1_NLP_Spark_Embeddings.ipynb` | Spark version is production |

**Reason**: Original approach didn't use Spark, later versions added Spark for scalability

---

## Dependencies & Technologies

### External Dependencies by Code Section

**ETL Scripts**:
- PySpark (SQL, DataFrames)
- SQLAlchemy
- psycopg2 (PostgreSQL)

**NLP Scripts**:
- PySpark (SQL, ML)
- spaCy (`en_core_web_sm`)
- Gensim (LDA)
- NLTK
- TextBlob
- Spark NLP
- fastText

**ML Scripts**:
- scikit-learn (LogReg, Tree, Forest, SVM)
- XGBoost
- PyCaret (AutoML)
- MLflow (tracking)
- pandas, numpy, scipy

**Version 1 Legacy**:
- MongoDB (pymongo) - no longer used

---

## Code Quality Issues (Known)

### From docs/hardcoded_paths_inventory.md:
- ✅ **Hardcoded paths**: 62+ instances across 12 files
  - EC2 paths: `/home/ubuntu/`
  - Local paths: `/home/jeff/`, `/home/jovyan/`
  - Need to migrate to `config.py`

### From docs/duplicate_code_analysis.md:
- ✅ **8 notebook/script pairs** with overlapping functionality
- ✅ **3 code patterns** duplicated across files

### From docs/deprecated_code_inventory.md:
- ✅ **version_0/**: Entirely obsolete (MongoDB)
- ✅ **version_1/**: Partially obsolete (AWS RDS)
- ❓ **Main notebooks**: Some may be outdated vs. scripts

---

## Refactoring Priorities

### Phase 2: Configuration Management (NEXT)
**Target**: All 13 production scripts in `src/`
1. Replace hardcoded paths with `config.py`
2. Remove AWS-specific code (RDS references)
3. Standardize database connection handling

### Phase 3: Code Consolidation
1. **Decide**: Keep notebooks or scripts as source of truth?
   - Option A: Scripts = production, notebooks = exploration only
   - Option B: Notebooks = single source, generate scripts from them
2. **Remove duplicates**: Eliminate Spark/non-Spark notebook duplicates
3. **Archive legacy**: Move version_0 and version_1 to archive folder

### Phase 4: Code Quality
1. Add docstrings to all functions
2. Implement logging (remove print statements)
3. Add error handling
4. Write unit tests
5. Apply code formatter (black)

### Phase 5: Pipeline Modernization
1. Decide on database strategy (PostgreSQL, Parquet, SQLite, or CSV-only)
2. Refactor ETL scripts
3. Refactor NLP scripts
4. Refactor ML scripts
5. Create end-to-end pipeline runner

---

## File Organization Recommendations

### Proposed Structure (After Refactoring)

```
src/
├── config.py                          # Configuration (KEEP)
├── etl/                               # ETL modules
│   ├── __init__.py
│   ├── spark_loader.py               # From 1_ETL_Spark.py
│   ├── data_combiner.py              # From 3_*.sql
│   └── prediction_adder.py           # From 11.1_*.py
├── nlp/                               # NLP modules
│   ├── __init__.py
│   ├── text_basic.py                 # From 2.1_*.py
│   ├── spacy_features.py             # From 2.2_*.py
│   ├── tfidf_models.py               # From 2.3.1_*.py
│   ├── embeddings.py                 # From 2.4.1_*.py + 2.4.2_*.py
│   └── topic_modeling.py             # From 2.5_*.py
├── modeling/                          # ML modules
│   ├── __init__.py
│   ├── base_models.py                # From 4.1_*.py
│   ├── dimensionality.py             # From 5_*.py
│   ├── classification.py             # From 6.1_*.py
│   └── regression.py                 # From 10.1_*.py
├── utils/                             # Utilities
│   ├── __init__.py
│   ├── database.py
│   ├── logging_config.py
│   └── validators.py
└── pipeline.py                        # Main orchestrator

notebooks/
├── exploration/                       # Keep EDA notebooks
│   ├── 3.2_EDA_Processed_Text_Data.ipynb
│   ├── non_text_eda.ipynb
│   └── 2.0_NLP_Wordclouds.ipynb
├── experiments/                       # Keep experimental notebooks
│   ├── 4.0_AutoML_PyCaret.ipynb
│   ├── 5.1_Feature_Selection.ipynb
│   ├── 7.0_REG_Target_Adjustments.ipynb
│   ├── 8.0_AutoML_PyCaret_Regression.ipynb
│   └── 9.1_Feature_Selection_Regression.ipynb
├── evaluation/                        # Keep evaluation notebooks
│   └── 12.1_Review_Ranking.ipynb
└── archive/                           # Archive outdated notebooks
    ├── version_0/
    ├── version_1/
    └── superseded/

tests/                                 # NEW - add tests
├── test_etl.py
├── test_nlp.py
└── test_modeling.py
```

---

## Summary Statistics

### Current Codebase
- **Production Scripts**: 13 files (4,813 lines)
- **Active Notebooks**: 19 files (~6MB total)
- **Legacy Version 1**: 4 main scripts + 6 helpers + 1 SQL file + 3 notebooks
- **Legacy Version 0**: 4 scripts (830 lines)
- **Total Files**: ~42 code files

### Code Distribution
- ETL: ~29% (1,373 lines)
- NLP: ~28% (1,331 lines)
- ML: ~44% (2,109 lines)
- Config: ~5% (256 lines)

### Known Issues
- 62+ hardcoded paths
- 8 notebook/script duplicates
- 2+ full legacy versions
- AWS dependencies (no longer available)
- No centralized logging
- No automated tests
- Inconsistent code style

### Refactoring Effort Estimate
- **Phase 2** (Config Migration): 12 files to update
- **Phase 3** (Consolidation): ~20 files to review/merge/archive
- **Phase 4** (Quality): 13 production files to enhance
- **Phase 5** (Modernization): Full pipeline redesign

---

## Quick Reference

### To Run Current Pipeline (If PostgreSQL Available)
```bash
# 1. ETL
python src/1_ETL_Spark.py

# 2. NLP (in order)
python src/2.1_NLP_Basic_Text_Processing_Spark.py
python src/2.2_NLP_Spacy_POS_ENT_DEP.py
python src/2.3.1_NLP_Spark_Tf-Idf_Models.py
python src/2.4.1_NLP_Spark_Text_Embeddings.py
python src/2.4.2_NLP_Fasttext.py
python src/2.5_NLP_Topic_Modeling_LDA.py

# 3. Combine features (SQL - run in PostgreSQL)
psql -f src/3_ETL_Combine_Processed_Text_Data.sql

# 4. Export to CSV (notebook)
# Run: notebooks/3.1_ETL_Combined_Data_to_CSV.ipynb

# 5. Train models
python src/4.1_ML_Base_Models.py
python src/6.1_ML_Logistic_Regression.py
python src/10.1_ML_Linear_Regression.py

# 6. Add predictions
python src/11.1_ETL_Add_Predictions.py
```

**Issue**: Steps 1-3 require PostgreSQL (not available without AWS RDS)

### Files That Can Run Standalone (No Database)
- ✅ Any modeling script with existing `train.csv`/`test.csv`
- ✅ `notebooks/3.2_EDA_Processed_Text_Data.ipynb` (EDA)
- ✅ `notebooks/4.0_AutoML_PyCaret.ipynb` (modeling experiments)
- ✅ All feature selection notebooks (if CSVs exist)
