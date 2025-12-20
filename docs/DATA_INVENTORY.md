# Data Inventory

**Generated**: December 19, 2025  
**Updated**: Post-cleanup and 2024 dataset download  
**Total Data Size**: ~11GB (2021) + 4GB zipped (2024)

---

## Summary

✅ **Clean Data Structure**: Reorganized and ready for refactoring!

### What You Have:
1. **Original 2021 Data**: All 5 Yelp JSON files (~11GB) - READY TO USE
2. **Original 2024 Data**: Yelp-JSON.zip (4GB compressed) - FOR LATER USE
3. **Models & Processed Data**: DELETED (will recreate during refactor)

### What You DON'T Have (Intentionally Cleaned):
- ❌ Intermediate processed data (deleted - will recreate)
- ❌ Trained models (deleted - will recreate)
- ❌ AWS RDS PostgreSQL database (no longer accessible)

---

## Detailed Inventory

### 1. Original 2021 Dataset (11GB)
**Location**: `data/original_json_2021/`

| File | Size | Records (approx) | Status |
|------|------|------------------|--------|
| yelp_academic_dataset_review.json | 6.5GB | 8,635,403 reviews | ✅ |
| yelp_academic_dataset_user.json | 3.4GB | 2,189,457 users | ✅ |
| yelp_academic_dataset_business.json | 119MB | 160,585 businesses | ✅ |
| yelp_academic_dataset_checkin.json | 380MB | 138,876 records | ✅ |
| yelp_academic_dataset_tip.json | 220MB | 1,162,119 tips | ✅ |

**Extra Files**:
- `yelp_business.csv` - 4.3MB (business_id + review_count)
- `yelp_reviews.csv` - 799MB (review data subset)

**Dataset Characteristics**:
- Date range: 2004-2020 (16 years)
- Geographic coverage: Multiple metro areas
- Primary focus: Restaurant/business reviews
- Use case: Training and refactoring base pipeline

---

### 2. Original 2024 Dataset (4GB compressed, ~8.65GB uncompressed)
**Location**: `data/original_json_2024/`
**Status**: Downloaded, kept zipped for future use

| File | Size | Status |
|------|------|--------|
| Yelp-JSON.zip | 4.0GB | ✅ Zipped |

**When Extracted Contains**:
- 5 JSON files (same structure as 2021)
- 1 PDF documentation
- Total uncompressed: ~8.65GB

**Dataset Characteristics** (per Yelp website):
- **Reviews**: 6,990,280 (19% fewer than 2021)
- **Businesses**: 150,346 (focused on 11 metro areas)
- **Photos**: 200,100 available (separate download)
- Date range: Updated through 2023-2024
- Changes: Cleaned/filtered dataset, focused geography

**Future Use**:
- Test refactored pipeline with newer data
- Compare model performance across dataset versions
- Potential for multimodal features (photos available)

**Extraction Instructions**:
```bash
cd data/original_json_2024/
unzip Yelp-JSON.zip
tar -xvf yelp_dataset.tar
```

---

### 3. Processed Data & Models
**Status**: ❌ DELETED (intentionally cleaned for fresh refactor)

**Previously Existed**:
- Model-ready CSVs (train.csv, test.csv) - 5.3GB
- Final predictions - 7.3GB  
- Trained models - 8GB

**Will Be Recreated During Refactoring**:
1. Intermediate NLP features
2. Model-ready train/test splits
3. Trained models (base + tuned)
4. Final predictions and rankings

---

## Data Pipeline Overview

### Original Pipeline (2020-2021)
```
JSON Files (2021 dataset)
    ↓
Spark ETL (1_ETL_Spark.py)
    ↓
PostgreSQL (AWS RDS - NO LONGER AVAILABLE)
    ↓
NLP Processing (Spark, spaCy, Gensim, FastText)
    ↓
Feature Engineering (SQL + Python)
    ↓
Model-Ready CSVs (train.csv, test.csv)
    ↓
ML Models (sklearn, XGBoost, PyCaret)
    ↓
Predictions & Rankings
```

### Refactored Pipeline (Future)
```
JSON Files (2021 dataset initially, 2024 later)
    ↓
Spark ETL (refactored, config-driven)
    ↓
Local Storage (Parquet/SQLite/PostgreSQL)
    ↓
NLP Processing (modernized, parallelized)
    ↓
Feature Engineering (centralized, documented)
    ↓
Model-Ready Data (versioned, validated)
    ↓
ML Models (reproducible, tracked)
    ↓
Predictions & Evaluation
```

---

## Missing Data (Acceptable Losses)

### 1. AWS PostgreSQL Database
**Status**: ❌ No longer accessible (account closed)
**Impact**: Low - can recreate from JSON files
**Contains**:
- Intermediate tables (text_data, text_spacy, etc.)
- Joined/processed data
- Temporary computations

### 2. Intermediate Processing Files
**Status**: ❌ Intentionally deleted
**Impact**: None - recreatable from source
**Examples**:
- *.pkl files (pickle intermediates)
- Temp CSV exports
- Debug outputs

---

## Storage Summary

```
data/
├── original_json_2021/          11 GB  [Source data - KEEP]
│   ├── yelp_academic_dataset_review.json     6.5 GB
│   ├── yelp_academic_dataset_user.json       3.4 GB
│   ├── yelp_academic_dataset_business.json   119 MB
│   ├── yelp_academic_dataset_checkin.json    380 MB
│   ├── yelp_academic_dataset_tip.json        220 MB
│   ├── yelp_business.csv                     4.3 MB
│   └── yelp_reviews.csv                      799 MB
│
└── original_json_2024/           4 GB  [Future use - KEEP]
    └── Yelp-JSON.zip                          4.0 GB
```

**Total Current Storage**: ~15 GB
**Previous Storage**: ~35 GB (before cleanup)
**Space Saved**: ~20 GB

---

## Data Schema Reference

### Review JSON Structure
| XGBoost | xgboost_sklearn_base_model_1M.joblib | Baseline classification |

**Training**: 1M sample, PowerTransformer + SMOTE

#### Final Models (4 models)
**Location**: `models/final_models/`

| Model | File | Purpose |
|-------|------|---------|
| LogReg (Top Features) | log_reg_cv_ALL_top_features_tuned.joblib | ✅ Primary classifier |
| LogReg (Submodels) | log_reg_cv_ALL_submodels_tuned.joblib | Alternative classifier |
| LinReg (Top Features) | lin_reg_ALL_top_features.joblib | ✅ Primary regressor |
| LinReg (Submodels) | lin_reg_ALL_submodels.joblib | Alternative regressor |

**Training**: Full 5.5M dataset, cross-validation, hyperparameter tuned

#### NLP Models (4 models)
**Location**: `models/nlp/`

| Model | Location | Size | Purpose |
|-------|----------|------|---------|
| FastText | fasttext_model_ALL | 240MB | Word embeddings → ft_prob |

```
{
    "review_id": "lWC-xP3rd6obsecCYsGZRg",
    "user_id": "ak0TdVmGKo4pwqdJSTLwWw",
    "business_id": "buF9druCkbuXLX526sGELQ",
    "stars": 4.0,
    "useful": 3,
    "funny": 1,
    "cool": 1,
    "text": "Review text here...",
    "date": "2014-10-11 03:34:02"
}
```

### User JSON Structure
```
{
    "user_id": "q_QQ5kBBwlCcbL1s4NVK3g",
    "name": "Jane",
    "review_count": 1220,
    "yelping_since": "2005-03-14 20:26:35",
    "useful": 15038,
    "funny": 10030,
    "cool": 11291,
    "elite": "2006,2007,2008,2009,2010,2011,2012,2013,2014",
    "friends": "xBDpTUbai0DXrvxCe3X16Q, ...",
    "fans": 1357,
    "average_stars": 3.85,
    "compliment_hot": 1710,
    "compliment_more": 163,
    ...
}
```

### Business JSON Structure
```
{
    "business_id": "6iYb2HFDywm3zjuRg0shjw",
    "name": "Oskar Blues Taproom",
    "address": "921 Pearl St",
    "city": "Boulder",
    "state": "CO",
    "postal_code": "80302",
    "latitude": 40.0175444,
    "longitude": -105.2833481,
    "stars": 4.0,
    "review_count": 86,
    "is_open": 1,
    "attributes": {...},
    "categories": "Gastropubs, Food, Beer Gardens, ...",
    "hours": {...}
}
```

---

## Next Steps for Refactoring

### Phase 2: Configuration Management
1. Update all scripts to use `src/config.py` for paths
2. Replace hardcoded paths in:
   - `1_ETL_Spark.py`
   - All `2.x_NLP_*.py` scripts
   - All `4.x-11.x_ML_*.py` scripts

### Phase 3: Database Strategy Decision
Choose one:
- **Option A**: CSV/Parquet-based (no database)
- **Option B**: Local PostgreSQL (replicate old approach)
- **Option C**: SQLite (lightweight alternative)

### Phase 4: Pipeline Refactoring
1. Refactor ETL scripts
2. Refactor NLP scripts
3. Refactor ML scripts
4. Add tests and validation

### Phase 5: Testing with New Data
1. Run refactored pipeline on 2021 data
2. Compare results with original
3. Test on 2024 data
4. Document differences

---

## Quick Reference

### File Counts
- JSON files (2021): 5
- JSON files (2024): 5 (zipped)
- Total records: 12.3M (2021), 8.3M (2024)

### Key Metrics (2021 Dataset)
- Reviews: 8,635,403
- Users: 2,189,457
- Businesses: 160,585
- Date span: 2004-2020 (16 years)

### Storage Impact of Cleanup
- Before: ~35 GB
- After: ~15 GB
- Savings: ~20 GB (57% reduction)

