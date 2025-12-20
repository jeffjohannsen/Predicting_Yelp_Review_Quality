# Hardcoded Paths Inventory

> **Generated**: December 19, 2025  
> **Purpose**: Catalog all hardcoded file paths throughout the codebase for refactoring

---

## Summary

**Total Hardcoded Path Instances**: 62+  
**Files Affected**: 12 source files  
**Environments**: EC2 (`/home/ubuntu/`), Local (`/home/jeff/`), Jupyter (`/home/jovyan/`)

---

## Priority 1: Active Source Files (src/)

### 1_ETL_Spark.py (4 hardcoded paths)

```python
Line 10:  "spark.driver.extraClassPath", "/home/ubuntu/postgresql-42.2.20.jar"
Line 20:  data_location = "/home/ubuntu/yelp_2021/data/"
```

**Impact**: HIGH - Entry point for entire pipeline  
**Fix**: Move to config for `SPARK_DRIVER_CLASSPATH`, `RAW_DATA_DIR`

---

### 2.1_NLP_Basic_Text_Processing_Spark.py (1 path)

```python
Line 15:  "spark.driver.extraClassPath", "/home/ubuntu/postgresql-42.2.20.jar"
```

**Impact**: HIGH - NLP pipeline start  
**Fix**: Use config for `SPARK_DRIVER_CLASSPATH`

---

### 2.2_NLP_Spacy_POS_ENT_DEP.py (0 paths)

No hardcoded paths! Uses database connections only.

---

### 2.3.1_NLP_Spark_Tf-Idf_Models.py (1 path)

```python
Line 44:  "spark.driver.extraClassPath", "/home/jovyan/postgresql-42.2.20.jar"
```

**Impact**: MEDIUM - Different environment (Jupyter)  
**Fix**: Use config for `SPARK_DRIVER_CLASSPATH`

---

### 2.4.1_NLP_Spark_Text_Embeddings.py (1 path)

```python
Line 52:  "spark.driver.extraClassPath", "/home/ubuntu/postgresql-42.2.22.jar"
```

**Impact**: MEDIUM  
**Fix**: Use config for `SPARK_DRIVER_CLASSPATH`

---

### 2.4.2_NLP_Fasttext.py (0 notable paths)

Uses database connections - minimal path issues

---

### 2.5_NLP_Topic_Modeling_LDA.py (0 notable paths)

Uses relative paths for model saving

---

### 4.1_ML_Base_Models.py (4 paths)

```python
Line 25:  data_location_ec2 = "/home/ubuntu/"
Line 26:  data_location_local = "/home/jeff/Documents/Data_Science_Projects/Yelp_Reviews/data/full_data/model_ready/"
Line 27:  filepath_prefix = data_location_local
Line 223: f"{filepath_prefix}train.csv"
Line 230: f"{filepath_prefix}test.csv"
```

**Pattern**: Has EC2/Local toggle but both are hardcoded  
**Impact**: HIGH - Used by all ML experiments  
**Fix**: Config variables for `TRAIN_DATA_PATH`, `TEST_DATA_PATH`

---

### 5_PCA_Dimensionality_Reduction.py (6 paths)

```python
Line 14:  filepath_prefix = "/home/ubuntu/"
Line 16:  # filepath_prefix = "/home/jeff/Documents/Data_Science_Projects/Yelp_Reviews/data/full_data/model_ready/"
Line 212: f"{filepath_prefix}train.csv"
Line 219: f"{filepath_prefix}test.csv"
Line 289: train_pca.to_csv(f"{filepath_prefix}train_pca.csv", index=False)
Line 290: test_pca.to_csv(f"{filepath_prefix}test_pca.csv", index=False)
```

**Pattern**: Commented-out local path  
**Impact**: MEDIUM - Optional dimensionality reduction  
**Fix**: Config for data paths

---

### 6.1_ML_Logistic_Regression.py (6 paths)

```python
Line 75:  data_location_ec2 = "/home/ubuntu/"
Line 76:  data_location_local = "/home/jeff/Documents/Data_Science_Projects/Yelp_Reviews/data/full_data/model_ready/"
Line 78:  filepath_prefix = data_location_local (if True)
Line 80:  filepath_prefix = data_location_ec2 (if False)
Line 273: f"{filepath_prefix}train.csv"
Line 280: f"{filepath_prefix}test.csv"
```

**Pattern**: Manual True/False toggle for environment  
**Impact**: HIGH - Primary classification model  
**Fix**: Auto-detect environment or use config

---

### 10.1_ML_Linear_Regression.py (6 paths)

```python
Line 74:  data_location_ec2 = "/home/ubuntu/"
Line 75:  data_location_local = "/home/jeff/Documents/Data_Science_Projects/Yelp_Reviews/data/full_data/model_ready/"
Line 77:  filepath_prefix = data_location_local (if True)
Line 79:  filepath_prefix = data_location_ec2 (if False)
Line 272: f"{filepath_prefix}train.csv"
Line 279: f"{filepath_prefix}test.csv"
```

**Pattern**: Same as 6.1 - manual toggle  
**Impact**: HIGH - Primary regression model  
**Fix**: Auto-detect environment or use config

---

### 11.1_ETL_Add_Predictions.py (10 paths)

```python
Line 34:  data_location_ec2 = "/home/ubuntu/"
Line 35:  data_location_local = "/home/jeff/Documents/Data_Science_Projects/Yelp_Reviews/data/full_data/model_ready/"
Line 37:  filepath_prefix = data_location_local (if True)
Line 39:  filepath_prefix = data_location_ec2 (if False)
Line 232: f"{filepath_prefix}train.csv"
Line 239: f"{filepath_prefix}test.csv"
Line 574: model_location = "/home/jeff/Documents/Data_Science_Projects/Yelp_Reviews/models/final_models/"
Line 597: filepath_prefix = "/home/jeff/Documents/Data_Science_Projects/Yelp_Reviews/data/full_data/final_predict/"
Line 599: # train.to_csv(f"{filepath_prefix}train_full.csv", index=False)
Line 600: # test.to_csv(f"{filepath_prefix}test_full.csv", index=False)
```

**Pattern**: Multiple path switches throughout  
**Impact**: HIGH - Generates final predictions  
**Fix**: Comprehensive config for data/model paths

---

## Priority 2: Legacy Code (version_0/, version_1/)

### version_1/python_scripts/migrate_db_2_aws.py (1 path)

```python
Line 52: f"/home/jeff/Documents/Galvanize_DSI/capstones/C2_Yelp_Review_Quality/data/full_data/yelp_2/{table_name}.csv"
```

**Impact**: LOW - Legacy migration script  
**Fix**: Archive entire version_0 and version_1 directories

---

## Path Pattern Analysis

### Pattern 1: EC2 vs Local Toggle (Most Common)
**Found in**: 4.1, 6.1, 10.1, 11.1

```python
# Current anti-pattern:
data_location_ec2 = "/home/ubuntu/"
data_location_local = "/home/jeff/Documents/Data_Science_Projects/Yelp_Reviews/data/full_data/model_ready/"
if True:  # <-- Manual toggle
    filepath_prefix = data_location_local
else:
    filepath_prefix = data_location_ec2
```

**Proposed solution**:
```python
# In config.py:
import os
from pathlib import Path

def get_environment():
    """Auto-detect environment based on hostname or env var"""
    if os.getenv('YELP_ENV'):
        return os.getenv('YELP_ENV')
    elif Path('/home/ubuntu').exists():
        return 'ec2'
    elif Path('/home/jovyan').exists():
        return 'jupyter'
    else:
        return 'local'

ENV = get_environment()

PATHS = {
    'ec2': {
        'data_dir': Path('/home/ubuntu/data'),
        'model_dir': Path('/home/ubuntu/models'),
        'jdbc_driver': Path('/home/ubuntu/postgresql-42.2.20.jar')
    },
    'local': {
        'data_dir': Path.home() / 'Documents/Data_Science/Yelp_Reviews/data',
        'model_dir': Path.home() / 'Documents/Data_Science/Yelp_Reviews/models',
        'jdbc_driver': Path('/usr/share/java/postgresql.jar')  # System location
    },
    'jupyter': {
        'data_dir': Path('/home/jovyan/work/data'),
        'model_dir': Path('/home/jovyan/work/models'),
        'jdbc_driver': Path('/home/jovyan/postgresql-42.2.20.jar')
    }
}

# Easy access:
TRAIN_DATA = PATHS[ENV]['data_dir'] / 'full_data/model_ready/train.csv'
TEST_DATA = PATHS[ENV]['data_dir'] / 'full_data/model_ready/test.csv'
MODEL_DIR = PATHS[ENV]['model_dir']
JDBC_DRIVER = PATHS[ENV]['jdbc_driver']
```

---

### Pattern 2: Spark Driver ClassPath (Spark scripts)
**Found in**: 1, 2.1, 2.3.1, 2.4.1

```python
# Current anti-pattern:
.config("spark.driver.extraClassPath", "/home/ubuntu/postgresql-42.2.20.jar")
```

**Proposed solution**:
```python
# In config.py:
SPARK_CONFIG = {
    'driver_classpath': str(PATHS[ENV]['jdbc_driver']),
    'master': 'local[7]',  # Also configurable
    'app_name': 'Yelp_ETL'  # Can be set per script
}

# In scripts:
from config import SPARK_CONFIG

spark = (ps.sql.SparkSession.builder
    .appName(SPARK_CONFIG['app_name'])
    .config("spark.driver.extraClassPath", SPARK_CONFIG['driver_classpath'])
    .master(SPARK_CONFIG['master'])
    .getOrCreate())
```

---

### Pattern 3: Commented-out paths (Many files)
**Found in**: 5, 11.1, and notebooks

These indicate:
- Experimental code
- Different execution environments
- Data export steps that were run once

**Fix**: Remove commented code after config is in place

---

## Recommended Refactoring Strategy

### Phase 1: Create config.py
1. Add environment auto-detection
2. Define all path constants
3. Add database configuration (reference confidential.py)
4. Add Spark configuration

### Phase 2: Update scripts (in order)
1. **Start with**: `4.1_ML_Base_Models.py` (simplest pattern)
2. **Then**: `6.1`, `10.1` (same pattern)
3. **Then**: `11.1_ETL_Add_Predictions.py` (multiple patterns)
4. **Then**: `1_ETL_Spark.py` (critical but straightforward)
5. **Finally**: NLP scripts (2.x)

### Phase 3: Remove dead code
1. Delete all commented-out path definitions
2. Remove manual True/False environment toggles
3. Archive version_0 and version_1 directories

### Phase 4: Documentation
1. Update README with environment setup
2. Document config.py usage
3. Add .env.example for local customization

---

## Files Requiring No Changes

✅ `2.2_NLP_Spacy_POS_ENT_DEP.py` - Already uses database only  
✅ `2.4.2_NLP_Fasttext.py` - Minimal path usage  
✅ `2.5_NLP_Topic_Modeling_LDA.py` - Uses relative paths

---

## Total Refactoring Impact

**Files to modify**: 8 core scripts  
**Lines to change**: ~30-40 lines across all files  
**Estimated effort**: 2-3 hours  
**Risk level**: LOW (adding config, not changing logic)  
**Testing**: Can test locally first, then verify on EC2

---

## Next Steps

1. ✅ Document all paths (this file)
2. ⏭️ Create `src/config.py` with environment detection
3. ⏭️ Update one script at a time
4. ⏭️ Test each update locally
5. ⏭️ Remove commented code
6. ⏭️ Archive version_0 and version_1
