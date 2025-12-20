# Duplicate Code Analysis

> **Generated**: December 19, 2025  
> **Purpose**: Identify duplicate functionality between notebooks and .py scripts

---

## Summary

**Notebooks with Matching Scripts**: 8 pairs  
**Pattern**: Notebooks used for development/exploration → scripts for production  
**Recommendation**: Keep notebooks for documentation, use scripts for execution

---

## Exact Duplicates (Notebook → Script)

### 1. ETL Pipeline

**Notebook**: `notebooks/1_ETL_Spark.ipynb`  
**Script**: `src/1_ETL_Spark.py`

**Duplicate Content**:
- Spark session setup
- JSON file loading (checkin, user, business, review)
- Data transformations (SQL queries)
- Writing to PostgreSQL

**Differences**:
- Notebook has exploration/visualization cells
- Notebook uses `/home/jovyan/` paths (Jupyter)
- Script uses `/home/ubuntu/` paths (EC2)

**Status**: ⚠️ NEARLY IDENTICAL - Script is production version

**Recommendation**: 
- Keep notebook for documentation/learning
- Use script for actual pipeline execution
- Add note at top of notebook: "See src/1_ETL_Spark.py for production version"

---

### 2. Basic NLP Text Processing

**Notebook**: `notebooks/2.1_NLP_Spark_Text_Basic.ipynb`  
**Script**: `src/2.1_NLP_Basic_Text_Processing_Spark.py`

**Duplicate Content**:
- Spark session creation
- PostgreSQL data loading
- Basic text features (token count, readability, sentiment)
- Writing results back to PostgreSQL

**Differences**:
- Notebook has EDA/visualization of features
- Different Spark driver paths

**Status**: ⚠️ NEARLY IDENTICAL

**Recommendation**: Archive notebook or repurpose for EDA only

---

### 3. spaCy Linguistic Features

**Notebook**: `notebooks/2.2_NLP_Spacy_POS_ENT_DEP.ipynb`  
**Script**: `src/2.2_NLP_Spacy_POS_ENT_DEP.py`

**Duplicate Content**:
- spaCy model loading
- POS, NER, Dependency extraction
- Chunked processing pattern
- PostgreSQL updates

**Differences**:
- Notebook has examples and visualizations
- Script is production-optimized (chunked processing)

**Status**: ⚠️ NEARLY IDENTICAL

**Recommendation**: Keep notebook for spaCy examples/documentation

---

### 4. TF-IDF + NB/SVM Models

**Notebook**: `notebooks/2.3.1_NLP_Spark_Tf-Idf_Models.ipynb`  
**Script**: `src/2.3.1_NLP_Spark_Tf-Idf_Models.py`

**Duplicate Content**:
- Spark ML Pipeline (TF-IDF → Naive Bayes/SVM)
- Model training and evaluation
- Prediction generation
- Saving results to database

**Differences**:
- Notebook has model comparison visualizations
- Different environments (jovyan vs ubuntu)

**Status**: ⚠️ NEARLY IDENTICAL

**Recommendation**: Script for execution, notebook for analysis

---

### 5. Word Embeddings (Spark NLP)

**Notebook**: `notebooks/2.4.1_NLP_Spark_Embeddings.ipynb`  
**Script**: `src/2.4.1_NLP_Spark_Text_Embeddings.py`

**Duplicate Content**:
- Spark NLP setup
- Word2Vec embeddings
- BERT/ELMo configurations (commented out)
- Model training

**Differences**:
- Notebook has embedding exploration
- Different experimental configurations

**Status**: ⚠️ NEARLY IDENTICAL

**Recommendation**: Consolidate - pick one approach

---

### 6. Topic Modeling (LDA)

**Notebook**: `notebooks/2.5_NLP_Topic_Modeling_LDA.ipynb`  
**Script**: `src/2.5_NLP_Topic_Modeling_LDA.py`

**Duplicate Content**:
- Text preprocessing (lemmatization, stopwords)
- LDA model training with Gensim
- Topic assignment to reviews
- Model saving

**Differences**:
- Notebook has topic visualization (pyLDAvis)
- Notebook explores different topic counts

**Status**: ⚠️ NEARLY IDENTICAL

**Recommendation**: Use script for training, notebook for topic exploration

---

### 7. PCA Dimensionality Reduction

**Notebook**: `notebooks/5.0_PCA_Dimensionality_Reduction.ipynb`  
**Script**: `src/5_PCA_Dimensionality_Reduction.py`

**Duplicate Content**:
- Data loading with dtypes
- StandardScaler preprocessing
- PCA fitting and transformation
- Variance explained analysis
- Saving PCA-transformed datasets

**Differences**:
- Notebook has scree plots and variance visualizations
- Notebook experiments with different n_components

**Status**: ⚠️ NEARLY IDENTICAL

**Recommendation**: Script for execution, notebook for PCA selection/visualization

---

### 8. Combined Data Export

**Notebook**: `notebooks/3.1_ETL_Combined_Data_to_CSV.ipynb`  
**Script**: N/A (SQL file: `src/3_ETL_Combine_Processed_Text_Data.sql`)

**Duplicate Content**:
- SQL queries to combine text + non-text data
- Export to CSV

**Status**: ⚠️ PARTIAL MATCH

**Recommendation**: SQL file contains query logic, notebook for execution

---

## Notebooks WITHOUT Matching Scripts

### Exploration/Analysis Only (Keep)

**1. `notebooks/2.0_NLP_Wordclouds.ipynb`**
- Purpose: Visualization only
- Status: ✅ UNIQUE - Keep for visualization examples

**2. `notebooks/3.2_EDA_Processed_Text_Data.ipynb`**
- Purpose: EDA on processed text features
- Status: ✅ UNIQUE - Keep for analysis

**3. `notebooks/non_text_eda.ipynb`**
- Purpose: EDA on user/business features
- Status: ✅ UNIQUE - Keep for analysis

**4. `notebooks/12.1_Review_Ranking.ipynb`**
- Purpose: Final review ranking and evaluation
- Status: ✅ UNIQUE - Keep as final analysis

---

### AutoML Experiments (Keep)

**5. `notebooks/4.0_AutoML_PyCaret.ipynb`**
- Purpose: PyCaret AutoML experiments
- Status: ✅ UNIQUE - Documents AutoML trials

**6. `notebooks/8.0_AutoML_PyCaret_Regression.ipynb`**
- Purpose: PyCaret regression experiments
- Status: ✅ UNIQUE - Documents AutoML trials

---

### Feature Engineering Experiments (Keep)

**7. `notebooks/5.1_Feature_Selection.ipynb`**
- Purpose: Feature selection experiments (classification)
- Status: ✅ UNIQUE - Documents feature importance

**8. `notebooks/9.1_Feature_Selection_Regression.ipynb`**
- Purpose: Feature selection experiments (regression)
- Status: ✅ UNIQUE - Documents feature importance

**9. `notebooks/7.0_REG_Target_Adjustments.ipynb`**
- Purpose: Target engineering for regression
- Status: ✅ UNIQUE - Documents target transformation

---

### Non-Spark Versions (Lower Priority)

**10. `notebooks/2.3_NLP_Tf-Idf_Models.ipynb`**
- Purpose: Non-Spark TF-IDF (sklearn)
- Status: ⚠️ SUPERSEDED by 2.3.1 (Spark version)
- Recommendation: Archive or note it's deprecated

**11. `notebooks/2.4_NLP_Embeddings.ipynb`**
- Purpose: Non-Spark embeddings
- Status: ⚠️ SUPERSEDED by 2.4.1 (Spark version)
- Recommendation: Archive or note it's deprecated

---

## Legacy Notebooks (version_1/)

**Located in**: `notebooks/version_1/`

**Files**:
- `01_Exploratory_Data_Analysis.ipynb`
- `02_Data_Prep_Pipeline.ipynb`
- `eda_prep.ipynb`

**Status**: 🗑️ DEPRECATED - Old version of pipeline

**Recommendation**: Archive entire `version_1/` directory

---

## Refactoring Recommendations

### Immediate Actions

1. **Add Headers to Duplicates** - Each notebook that has a matching script should have a header cell:
   ```markdown
   # ⚠️ PRODUCTION VERSION AVAILABLE
   This notebook was used for development and exploration.
   For production execution, use: `src/X.X_Script_Name.py`
   ```

2. **Organize Notebooks by Purpose**:
   ```
   notebooks/
   ├── production/          # Mirrors src/ (development versions)
   │   ├── 1_ETL_Spark.ipynb
   │   ├── 2.1_NLP_Basic.ipynb
   │   └── ...
   ├── analysis/            # EDA and exploration
   │   ├── 2.0_Wordclouds.ipynb
   │   ├── 3.2_EDA_Text.ipynb
   │   └── non_text_eda.ipynb
   ├── experiments/         # AutoML, feature selection
   │   ├── 4.0_AutoML_PyCaret.ipynb
   │   ├── 5.1_Feature_Selection.ipynb
   │   └── ...
   └── archive/             # Deprecated versions
       ├── 2.3_NLP_TfIdf.ipynb (non-Spark)
       └── version_1/
   ```

3. **Remove True Duplicates** - After reorganization:
   - Consider removing notebooks that are 100% duplicates
   - Keep only if they have unique visualizations/analysis
   - Add README.md in notebooks/ explaining structure

### Long-term Strategy

**Notebooks are for**:
- Documentation of approach
- Visualization and analysis
- Experimentation and prototyping
- Teaching/explaining methodology

**Scripts are for**:
- Production execution
- Automated pipelines
- Reproducible results
- CI/CD integration

---

## Duplicate Code Patterns (Within Scripts)

### Pattern 1: Data Loading Boilerplate

**Found in**: 4.1, 5.0, 6.1, 10.1, 11.1

```python
# This pattern repeats in EVERY ML script:
datatypes = {
    "target_reg": "int16",
    "review_stars": "int16",
    # ... 60+ lines of dtype definitions
}
train = pd.read_csv(f"{filepath_prefix}train.csv", nrows=X, dtype=datatypes)
test = pd.read_csv(f"{filepath_prefix}test.csv", nrows=Y, dtype=datatypes)
```

**Recommendation**: Create `src/utils/data_loader.py`:
```python
def load_train_test(train_rows=None, test_rows=None):
    """Load train and test data with predefined dtypes"""
    # Define dtypes once
    # Load and return both datasets
```

---

### Pattern 2: Base Model Process Class

**Found in**: 4.1, 6.1, 10.1

Three nearly identical classes:
- `Base_Model_Process` (4.1) - for classification
- `Base_Model_Process` (6.1) - for classification with MLflow
- `Base_Model_Process_Regression` (10.1) - for regression

**Recommendation**: Consolidate into `src/utils/model_utils.py`:
```python
class ModelTrainer:
    def __init__(self, model, model_name, task='classification'):
        # Unified class for both classification and regression
```

---

### Pattern 3: Spark Session Setup

**Found in**: 1, 2.1, 2.3.1, 2.4.1

```python
# Nearly identical in every Spark script:
spark = (ps.sql.SparkSession.builder
    .appName("Yelp_XXX")
    .config("spark.driver.extraClassPath", "/path/to/jar")
    .master('local[7]')
    .getOrCreate())
```

**Recommendation**: Create `src/utils/spark_utils.py`:
```python
def get_spark_session(app_name, config_overrides=None):
    """Create configured Spark session"""
```

---

## Summary Statistics

| Category | Count | Action |
|----------|-------|--------|
| Notebooks with matching scripts | 8 | Add headers, reorganize |
| Unique analysis notebooks | 9 | Keep as-is |
| Deprecated notebooks | 5 | Archive to `notebooks/archive/` |
| Duplicate code patterns in scripts | 3 | Refactor to utils/ |
| Version 1 legacy notebooks | 3 | Archive |

**Total Cleanup Impact**:
- Can reorganize 20+ notebooks
- Can create 3-4 utility modules
- Eliminate ~500+ lines of duplicate code
