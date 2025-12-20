# Deprecated and Legacy Code Inventory

> **Generated**: December 19, 2025  
> **Purpose**: Identify unused, outdated, and legacy code for archival or removal

---

## Summary

**Legacy Directories**: 2 (version_0, version_1)  
**Deprecated Notebooks**: 5  
**Total Files to Archive**: 15+  
**Estimated Space to Reclaim**: Unknown (mostly small .py files)

---

## Priority 1: Legacy Directories (ARCHIVE ENTIRE)

### src/version_0/

**Location**: `src/version_0/`

**Files**:
1. `y1_json2mongo.py` - MongoDB migration (old approach)
2. `y1_model_prep.py` - Old model preparation
3. `y1_mongo2sql.py` - MongoDB to SQL migration
4. `y1_pipeline.py` - Version 0 pipeline

**Technology Used**: MongoDB (now uses PostgreSQL)

**Status**: 🗑️ COMPLETELY DEPRECATED

**Why Deprecated**:
- Project moved from MongoDB → PostgreSQL
- Entire ETL approach was redesigned in version 1
- Uses outdated file paths
- No longer compatible with current pipeline

**Action**: 
- Move to `src/archive/version_0/`
- Add README.md: "Historical version - used MongoDB. See src/1_ETL_Spark.py for current approach"

---

### src/version_1/

**Location**: `src/version_1/`

**Files**:

**Python Scripts**:
1. `a1_Model_Pipeline.py` - Old model pipeline
2. `a2_NLP.py` - Old NLP processing
3. `a3_Random_Forest_Metrics.py` - Old RF evaluation
4. `a4_Model_Setup.py` - Old model configuration

**Subdirectories**:
- `python_scripts/` - Contains `migrate_db_2_aws.py` and others
- `sql/` - Old SQL queries

**Status**: ⚠️ PARTIALLY DEPRECATED

**Why Deprecated**:
- Replaced by numbered scripts (1.x, 2.x, 4.x, etc.)
- Different organization structure
- Some contain hardcoded old paths (`/home/jeff/Documents/Galvanize_DSI/capstones/`)

**Evidence from `migrate_db_2_aws.py`**:
```python
Line 52: f"/home/jeff/Documents/Galvanize_DSI/capstones/C2_Yelp_Review_Quality/data/full_data/yelp_2/..."
```
- References "Galvanize_DSI/capstones" (old location)
- Now project is at `/home/jeff/Documents/Data_Science/Yelp_Reviews/`

**Action**:
- Review `sql/` directory for potentially useful queries
- Archive entire `version_1/` to `src/archive/version_1/`
- Add README.md explaining deprecation

---

## Priority 2: Deprecated Notebooks

### notebooks/version_1/

**Location**: `notebooks/version_1/`

**Files**:
1. `01_Exploratory_Data_Analysis.ipynb`
2. `02_Data_Prep_Pipeline.ipynb`
3. `eda_prep.ipynb`

**Status**: 🗑️ COMPLETELY DEPRECATED

**Why Deprecated**:
- Old version of pipeline
- Replaced by current numbered notebooks (1.x, 2.x, etc.)
- Likely incompatible with current data structure

**Action**:
- Move to `notebooks/archive/version_1/`
- These are already in a version_1 subdirectory, so clearly deprecated

---

### Non-Spark NLP Notebooks (Superseded)

#### notebooks/2.3_NLP_Tf-Idf_Models.ipynb

**Status**: ⚠️ SUPERSEDED by `2.3.1_NLP_Spark_Tf-Idf_Models.ipynb`

**Why Deprecated**:
- Non-Spark version (uses sklearn only)
- Project standardized on Spark for scalability
- 2.3.1 (Spark version) is used in production pipeline

**Action**:
- Move to `notebooks/archive/non_spark_versions/`
- Add note: "See 2.3.1 for Spark-based production version"

---

#### notebooks/2.4_NLP_Embeddings.ipynb

**Status**: ⚠️ SUPERSEDED by `2.4.1_NLP_Spark_Embeddings.ipynb`

**Why Deprecated**:
- Non-Spark version
- 2.4.1 uses Spark NLP library
- Not part of production pipeline

**Action**:
- Move to `notebooks/archive/non_spark_versions/`
- Add note: "See 2.4.1 for Spark-based production version"

---

## Priority 3: Commented-Out Code in Active Files

### src/11.1_ETL_Add_Predictions.py

**Lines 599-600**:
```python
# train.to_csv(f"{filepath_prefix}train_full.csv", index=False)
# test.to_csv(f"{filepath_prefix}test_full.csv", index=False)
```

**Why Commented**: Data export already completed, don't need to re-run

**Action**: Remove after confirming files exist in `data/full_data/final_predict/`

---

### src/2.3.1_NLP_Spark_Tf-Idf_Models.py

**Lines 194-200**:
```python
# train_finished.write.jdbc(...)
# test_finished.write.jdbc(...)
```

**Why Commented**: Alternative approach, not used

**Action**: Remove or move to documentation comment

---

### src/2.5_NLP_Topic_Modeling_LDA.py

**Lines 73, 77**:
```python
# pkl.dump(processed_train, f)
# pkl.dump(processed_test, f)
```

**Why Commented**: Pickle saving step already completed

**Action**: Remove after confirming .pkl files exist

**Line 129**:
```python
# final_lda_model = gensim.models.LdaMulticore.load('LDA_Model_1M', mmap='r')
```

**Why Commented**: Alternative loading method

**Action**: Remove or document as alternative

---

### src/2.4.1_NLP_Spark_Text_Embeddings.py

**Lines 395-403**: Commented BERT/ELMO configurations

```python
# bert_embeddings = (BertEmbeddings.pretrained()...
# elmo_embeddings = (ElmoEmbeddings.pretrained()...
# bse = (BertSentenceEmbeddings.pretrained()...
```

**Why Commented**: Experimental approaches not used in final pipeline

**Action**: Remove or move to separate experimental script

---

## Priority 4: Duplicate Model Info

### models/version_1/model_info.csv

**File**: `models/version_1/model_info.csv`

**Size**: 8533 rows (experiment tracking)

**Status**: ⚠️ LEGACY EXPERIMENT DATA

**Why Keep**: Historical record of 8533 model experiments

**Why Archive**: Version 1 experiments, current work uses MLflow

**Action**: 
- Keep for historical reference
- Move to `models/archive/version_1/`
- Document that current work uses MLflow, not CSV tracking

---

## Priority 5: Dead Code Patterns

### Manual Environment Toggles

**Found in**: 4.1, 6.1, 10.1, 11.1

```python
if True:  # <-- Manual toggle
    filepath_prefix = data_location_local
else:
    filepath_prefix = data_location_ec2
```

**Action**: Remove after config.py is implemented (Phase 2)

---

### Commented Path Alternatives

**Found in**: 5.0, 11.1

```python
# filepath_prefix = "/home/ubuntu/"
# filepath_prefix = "/home/jeff/Documents/..."  # Commented alternative
```

**Action**: Remove all commented paths after config.py implementation

---

## Archive Directory Structure (Proposed)

```
src/archive/
├── version_0/
│   ├── README.md (explains MongoDB approach)
│   ├── y1_json2mongo.py
│   ├── y1_model_prep.py
│   ├── y1_mongo2sql.py
│   └── y1_pipeline.py
└── version_1/
    ├── README.md (explains old structure)
    ├── a1_Model_Pipeline.py
    ├── a2_NLP.py
    ├── a3_Random_Forest_Metrics.py
    ├── a4_Model_Setup.py
    ├── python_scripts/
    └── sql/

notebooks/archive/
├── version_1/
│   ├── README.md
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Data_Prep_Pipeline.ipynb
│   └── eda_prep.ipynb
└── non_spark_versions/
    ├── README.md (explains why superseded)
    ├── 2.3_NLP_Tf-Idf_Models.ipynb
    └── 2.4_NLP_Embeddings.ipynb

models/archive/
└── version_1/
    ├── README.md (explains CSV tracking vs MLflow)
    └── model_info.csv
```

---

## Archival Checklist

### Phase 1: Create Archive Structure
- [ ] Create `src/archive/version_0/`
- [ ] Create `src/archive/version_1/`
- [ ] Create `notebooks/archive/version_1/`
- [ ] Create `notebooks/archive/non_spark_versions/`
- [ ] Create `models/archive/version_1/`

### Phase 2: Move Files
- [ ] Move `src/version_0/*` → `src/archive/version_0/`
- [ ] Move `src/version_1/*` → `src/archive/version_1/`
- [ ] Move `notebooks/version_1/*` → `notebooks/archive/version_1/`
- [ ] Move non-Spark notebooks → `notebooks/archive/non_spark_versions/`
- [ ] Move `models/version_1/model_info.csv` → `models/archive/version_1/`

### Phase 3: Add Documentation
- [ ] Create README.md in each archive directory explaining:
  - What was archived
  - Why it was archived
  - When it was archived
  - What replaced it

### Phase 4: Remove Commented Code
- [ ] Remove commented paths after config.py implementation
- [ ] Remove commented data export lines (after verification)
- [ ] Remove manual environment toggles
- [ ] Remove experimental BERT/ELMO code (or move to experiments/)

### Phase 5: Update Documentation
- [ ] Update main README.md to remove references to version_0/version_1
- [ ] Update file_dependencies.md if needed
- [ ] Add note about archived code in CONTRIBUTING.md

---

## Benefits of Archival

1. **Cleaner Codebase**: Remove confusion about which code to use
2. **Easier Navigation**: Fewer directories and files to search through
3. **Clear History**: Archived code documents evolution of project
4. **Reduced Maintenance**: Don't need to update deprecated code
5. **Better Onboarding**: New developers see current approach only

---

## Safety Measures

**Before archiving**:
1. ✅ Verify all archived functionality is replaced in current code
2. ✅ Git commit current state (backup before archiving)
3. ✅ Test current pipeline still works
4. ✅ Create archive/ directories with README.md files first
5. ✅ Move files (don't delete) - can always retrieve from archive/

**Git practices**:
- Don't delete from git history (files still accessible in old commits)
- Use `git mv` to preserve history when moving files
- Tag commit before archival: `git tag pre-archival-cleanup`

---

## Timeline

**Recommended approach**: Do this AFTER Phase 2 (Configuration Management) is complete

**Reason**: Once config.py is working, can safely remove all commented paths and environment toggles, making archival cleaner

**Estimated time**: 1-2 hours for complete archival + documentation
