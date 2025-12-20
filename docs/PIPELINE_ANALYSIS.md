# Pipeline Analysis - Src vs Notebooks

## Pipeline Flow Overview

The Yelp Review Quality Prediction pipeline follows this flow:

```
1. ETL (Extract, Transform, Load)
   JSON → Spark Processing → PostgreSQL (legacy) or CSV

2. NLP Processing
   Text → Basic Features → Linguistic Features → Topic Models → Embeddings → Predictions

3. Feature Combination
   All NLP outputs + Metadata → Combined dataset

4. Machine Learning
   Combined Data → Base Models → Feature Selection/PCA → Tuned Models → Predictions

5. Evaluation & Ranking
   Predictions → Review Ranking → Performance Analysis
```

## Stage-by-Stage Comparison

### Stage 1: ETL (Extract, Transform, Load)

**Src Files:**
- `1_ETL_Spark.py` (337 lines)
  - Loads 5 JSON files (review, user, business, checkin, tip)
  - Spark SQL transformations
  - Creates combined dataset
  - Splits into train/test/holdout
  - Saves to PostgreSQL (AWS RDS - NO LONGER AVAILABLE)

**Notebooks:**
- `1_ETL_Spark.ipynb` (74 KB)
  - Interactive version of the script
  - Likely contains exploration and validation
  - **Purpose**: Development/prototyping

**Status**: 🟡 DUPLICATE - Notebook for exploration, script for production
**Decision Needed**: Keep notebook for reference/visualization, use script as source of truth

---

### Stage 2: NLP Processing (Text Features)

#### 2.0: Visualization

**Src Files:** None

**Notebooks:**
- `2.0_NLP_Wordclouds.ipynb` (8.8 KB)
  - Text visualization
  - Word clouds for quality/not-quality reviews

**Status**: ✅ KEEP NOTEBOOK ONLY - Visualization work, not production code

---

#### 2.1: Basic Text Features

**Src Files:**
- `2.1_NLP_Basic_Text_Processing_Spark.py` (158 lines)
  - Basic counts (char, word, sentence)
  - Readability scores (Flesch-Kincaid, ARI)
  - Sentiment (polarity, subjectivity) via TextBlob
  - Uses Spark for distributed processing
  - Saves to PostgreSQL

**Notebooks:**
- `2.1_NLP_Spark_Text_Basic.ipynb` (20 KB)
  - Interactive version of script
  - Development/testing

**Status**: 🟡 DUPLICATE - Notebook for exploration, script for production
**Decision Needed**: Archive notebook or keep for validation

---

#### 2.2: Linguistic Features (spaCy)

**Src Files:**
- `2.2_NLP_Spacy_POS_ENT_DEP.py` (237 lines)
  - Part-of-speech percentages (17 types: ADJ, NOUN, VERB, etc.)
  - Named entity percentages (18 types: PERSON, ORG, GPE, etc.)
  - Dependency percentages (45 types: nsubj, dobj, etc.)
  - Uses spaCy en_core_web_sm
  - Processes in chunks (100K records)
  - Saves to PostgreSQL

**Notebooks:**
- `2.2_NLP_Spacy_POS_ENT_DEP.ipynb` (104 KB)
  - Interactive version
  - May include linguistic feature exploration

**Status**: 🟡 DUPLICATE - Notebook for exploration, script for production

---

#### 2.3: TF-IDF + Classification Models

**Src Files:**
- `2.3.1_NLP_Spark_Tf-Idf_Models.py` (206 lines)
  - TF-IDF vectorization with Spark
  - Naive Bayes classifier
  - SVM classifier
  - Generates probability predictions as features
  - Spark version (distributed)

**Notebooks:**
- `2.3_NLP_Tf-Idf_Models.ipynb` (70 KB)
  - Non-Spark version (likely smaller dataset)
  - Model development/testing
- `2.3.1_NLP_Spark_Tf-Idf_Models.ipynb` (81 KB)
  - Spark version (matches script)

**Status**: 🔴 DUPLICATES - Two notebooks + one script doing the same thing
**Decision Needed**: Keep one notebook for exploration, delete the other

---

#### 2.4: Word Embeddings

**Src Files:**
- `2.4.1_NLP_Spark_Text_Embeddings.py` (336 lines)
  - Spark NLP word embeddings
  - Classification models on embeddings
  - Spark version
- `2.4.2_NLP_Fasttext.py` (84 lines)
  - FastText embeddings
  - Separate approach

**Notebooks:**
- `2.4_NLP_Embeddings.ipynb` (50 KB)
  - Non-Spark embeddings (development)
- `2.4.1_NLP_Spark_Embeddings.ipynb` (101 KB)
  - Spark version (matches script)

**Status**: 🔴 DUPLICATES - Multiple approaches (Spark NLP, FastText, non-Spark)
**Decision Needed**: Consolidate or clarify which embedding approach is final

---

#### 2.5: Topic Modeling (LDA)

**Src Files:**
- `2.5_NLP_Topic_Modeling_LDA.py` (181 lines)
  - Gensim LDA topic modeling
  - 5 topics
  - Generates topic probabilities as features
  - Saves to PostgreSQL

**Notebooks:**
- `2.5_NLP_Topic_Modeling_LDA.ipynb` (1.5 MB)
  - Interactive LDA development
  - Likely includes topic visualization

**Status**: 🟡 DUPLICATE - Notebook for topic exploration/viz, script for production

---

### Stage 3: Feature Combination

**Src Files:**
- `3_ETL_Combine_Processed_Text_Data.sql` (401 lines)
  - Massive SQL query
  - Joins all NLP feature tables
  - Creates combined dataset with 200+ features
  - PostgreSQL-specific

**Notebooks:**
- `3.1_ETL_Combined_Data_to_CSV.ipynb` (25 KB)
  - Exports combined data from PostgreSQL to CSV
  - Utility notebook
- `3.2_EDA_Processed_Text_Data.ipynb` (2.9 MB - LARGE!)
  - EDA on the processed text features
  - Likely heavy visualizations

**Status**: 
- SQL script: ✅ PRODUCTION CODE
- 3.1 notebook: 🟡 UTILITY - May be one-time use
- 3.2 notebook: ✅ KEEP - EDA/visualization work

---

### Stage 4: Machine Learning

#### 4.0: AutoML Exploration

**Src Files:** None

**Notebooks:**
- `4.0_AutoML_PyCaret.ipynb` (152 KB)
  - PyCaret AutoML experiments
  - Model comparison
  - Classification task

**Status**: ✅ KEEP NOTEBOOK - Exploration/experimentation, not production

---

#### 4.1: Base Models

**Src Files:**
- `4.1_ML_Base_Models.py` (366 lines)
  - Trains 4 base models on 1M records
  - LogisticRegression, DecisionTree, RandomForest, XGBoost
  - Saves models to `models/base_models/`
  - Uses MLflow tracking

**Notebooks:** None directly matching

**Status**: ✅ KEEP SCRIPT - Production model training

---

#### 5.x: Dimensionality Reduction & Feature Selection

**Src Files:**
- `5_PCA_Dimensionality_Reduction.py` (290 lines)
  - PCA for reducing feature space
  - Tests multiple n_components
  - Saves PCA-transformed datasets

**Notebooks:**
- `5.0_PCA_Dimensionality_Reduction.ipynb` (141 KB)
  - Interactive PCA exploration
  - Likely includes scree plots, variance explained
- `5.1_Feature_Selection.ipynb` (447 KB)
  - Feature selection techniques
  - Feature importance analysis
  - Classification focus

**Status**: 
- Script: ✅ PRODUCTION
- Notebooks: ✅ KEEP - Exploration/visualization of dimensionality reduction

---

#### 6.1: Logistic Regression (Tuned)

**Src Files:**
- `6.1_ML_Logistic_Regression.py` (691 lines)
  - GridSearchCV for hyperparameter tuning
  - Cross-validation
  - Full dataset training
  - Saves tuned models

**Notebooks:** None directly matching

**Status**: ✅ KEEP SCRIPT - Production model training

---

#### 7.0-9.x: Regression Tasks

**Src Files:**
- `10.1_ML_Linear_Regression.py` (653 lines)
  - Linear regression models
  - Continuous target (vote count prediction)

**Notebooks:**
- `7.0_REG_Target_Adjustments.ipynb` (116 KB)
  - Target engineering for regression
  - Scaling/transformation exploration
- `8.0_AutoML_PyCaret_Regression.ipynb` (287 KB)
  - PyCaret AutoML for regression
  - Model comparison
- `9.1_Feature_Selection_Regression.ipynb` (608 KB)
  - Feature selection for regression task
  - Feature importance

**Status**:
- Script: ✅ PRODUCTION
- Notebooks: ✅ KEEP - Exploration/experimentation

---

#### 11.1: Add Predictions

**Src Files:**
- `11.1_ETL_Add_Predictions.py` (600 lines)
  - Loads trained models
  - Generates predictions on test set
  - Adds predictions as features
  - Saves augmented dataset

**Notebooks:** None

**Status**: ✅ KEEP SCRIPT - Production pipeline step

---

### Stage 5: Evaluation & Ranking

**Src Files:** None

**Notebooks:**
- `12.1_Review_Ranking.ipynb` (128 KB)
  - Review ranking system
  - Compares model predictions
  - Evaluation metrics
  - Final output demonstration

**Status**: ✅ KEEP NOTEBOOK - Results analysis and visualization

---

### Additional Notebooks

**Notebooks:**
- `non_text_eda.ipynb` (1.6 MB)
  - EDA on non-text features (user, business metadata)
  - Likely early exploratory work

**Status**: ✅ KEEP - EDA/exploration work

---

## Summary Statistics

### Src Files (13 files, 4,813 lines)
- **ETL**: 3 files (1,373 lines)
- **NLP**: 6 files (1,331 lines)
- **ML**: 4 files (2,109 lines)

### Notebooks (19 files)
- **ETL**: 3 notebooks (74 KB + 25 KB + 2.9 MB)
- **NLP**: 9 notebooks (~2 MB total)
- **ML**: 6 notebooks (~1.9 MB total)
- **Other**: 1 notebook (1.6 MB EDA)

### Duplicate Categories

1. **🔴 Full Duplicates** (Script + Notebook doing identical work):
   - ETL: `1_ETL_Spark` (script + notebook)
   - NLP Basic: `2.1_NLP_Basic` (script + notebook)
   - NLP spaCy: `2.2_NLP_Spacy` (script + notebook)
   - NLP TF-IDF: `2.3.1_NLP_TF-IDF` (script + 2 notebooks!)
   - NLP Embeddings: `2.4.x` (2 scripts + 2 notebooks!)
   - NLP LDA: `2.5_NLP_LDA` (script + notebook)

2. **✅ Notebooks-Only** (Visualization/Exploration):
   - Wordclouds (2.0)
   - AutoML (4.0, 8.0)
   - Feature Selection (5.1, 9.1)
   - Target Adjustments (7.0)
   - Review Ranking (12.1)
   - Non-text EDA

3. **✅ Scripts-Only** (Production):
   - Base Models (4.1)
   - Tuned LogReg (6.1)
   - Linear Regression (10.1)
   - Add Predictions (11.1)
   - Config (config.py)

---

## Key Issues Identified

### 1. PostgreSQL Dependency (NO LONGER AVAILABLE)
**ALL scripts save to PostgreSQL on AWS RDS** - database no longer exists!
- Scripts will fail when trying to save results
- Need to refactor to save to CSV/Parquet instead

### 2. Duplicate NLP Notebooks
- Multiple versions of same processing (Spark vs non-Spark)
- Likely developed on different dataset sizes
- Need to decide: keep one for reference or archive all

### 3. Missing Time Discounting
- Confirmed: Current scripts do NOT do time discounting
- `archive/version_1/eda_prep.py` has the logic
- **CRITICAL**: Must implement before re-running pipeline

### 4. Notebook/Script Relationship Unclear
- Some notebooks are clearly development → script
- Others may have diverged
- Need to verify which is "source of truth"

---

## Recommendations for Pipeline Consolidation

### Phase 1: Classify Each File

For each duplicate (script + notebook):

**Option A: Script is Source of Truth**
- Archive notebook to `notebooks/archive/development/`
- Keep notebook if it has unique visualizations

**Option B: Notebook is Source of Truth**
- Extract production code from notebook → script
- Keep notebook for interactive exploration

**Option C: Both Serve Different Purposes**
- Script: Production (non-interactive)
- Notebook: Exploration/visualization/validation
- Keep both but document purposes

### Phase 2: PostgreSQL → CSV/Parquet Migration
- Update ALL scripts to save to local files instead of PostgreSQL
- Use Parquet for intermediate data (more efficient than CSV)
- Update file paths to use `config.py`

### Phase 3: Time Discounting Implementation
- Extract logic from `archive/version_1/eda_prep.py`
- Implement in refactored ETL pipeline
- Verify all TD features are created

### Phase 4: Organize Notebooks
```
notebooks/
  development/        # Original exploration notebooks (archive)
  visualization/      # EDA, wordclouds, charts
  experimentation/    # AutoML, feature selection trials
  validation/         # Results analysis, ranking
```

---

## Next Steps

1. **Walk through each pipeline stage** systematically
2. **Compare script vs notebook** for each duplicate
3. **Decide: Archive, Keep, or Merge** for each file
4. **Document purpose** of remaining files
5. **Create refactoring plan** for Phase 2 (config migration)

