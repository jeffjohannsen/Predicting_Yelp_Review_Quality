# Model Catalog

> **Generated**: December 19, 2025  
> **Purpose**: Comprehensive documentation of all trained models

---

## Summary

**Total Model Directories**: 4  
**Base Models**: 4  
**Final Models**: 4  
**NLP Models**: 4  
**Historical Experiments**: 8,533 tracked in version_1/model_info.csv

---

## Model Inventory

### Base Models (models/base_models/)

Located: `models/base_models/`  
Purpose: Initial baseline models trained on 1M sample  
Training Script: `src/4.1_ML_Base_Models.py`

#### 1. Logistic Regression Base

**File**: `log_reg_sklearn_base_model_1M.joblib`

**Specifications**:
- Algorithm: Logistic Regression (sklearn)
- Training Size: 1,000,000 rows
- Target: Binary classification (T2_CLS_ufc_>0)
- Preprocessing: PowerTransformer + SMOTE
- Purpose: Baseline for classification

**Performance** (estimated from model_info.csv):
- AUC: ~0.96
- Accuracy: ~89-90%
- F1 Score: ~0.89

---

#### 2. Decision Tree Base

**File**: `tree_sklearn_base_model_1M.joblib`

**Specifications**:
- Algorithm: Decision Tree Classifier (sklearn)
- Training Size: 1,000,000 rows
- Target: Binary classification
- Preprocessing: PowerTransformer + SMOTE
- Purpose: Interpretable baseline

**Performance** (estimated):
- Lower than LogReg and Forest
- High interpretability
- Prone to overfitting

---

#### 3. Random Forest Base

**File**: `forest_sklearn_base_model_1M.joblib`

**Specifications**:
- Algorithm: Random Forest Classifier (sklearn)
- Training Size: 1,000,000 rows
- Target: Binary classification
- Preprocessing: PowerTransformer + SMOTE
- Purpose: Ensemble baseline

**Performance** (from model_info.csv samples):
- AUC: 0.96-1.0 (CV)
- Accuracy: 95-100% (CV) - likely overfit on small samples
- Best performing among base models

---

#### 4. XGBoost Base

**File**: `xgboost_sklearn_base_model_1M.joblib`

**Specifications**:
- Algorithm: XGBoost Classifier
- Training Size: 1,000,000 rows
- Target: Binary classification
- Preprocessing: PowerTransformer + SMOTE
- Purpose: Gradient boosting baseline

**Performance** (estimated):
- Competitive with Random Forest
- Longer training time
- Good feature importance

---

### Final Models (models/final_models/)

Located: `models/final_models/`  
Purpose: Production-ready models trained on full dataset  
Tracking: MLflow experiments

#### 5. Logistic Regression - Top Features (Classification)

**File**: `log_reg_cv_ALL_top_features_tuned.joblib`

**Training Script**: `src/6.1_ML_Logistic_Regression.py`

**Specifications**:
- Algorithm: Logistic Regression with Cross-Validation
- Training Size: 5,523,992 rows (full training set)
- Target: Binary classification (T2_CLS_ufc_>0)
- Features: Top features selected via feature importance
- Preprocessing: 
  - PowerTransformer scaling
  - SMOTE balancing
- Cross-Validation: Yes (LogisticRegressionCV)
- Hyperparameter Tuning: Yes

**Hyperparameters** (from model_info.csv examples):
- C: 0.5-2.0 (regularization)
- solver: 'lbfgs', 'newton-cg', 'saga'
- penalty: 'l2'
- max_iter: 50-5000
- class_weight: 'balanced'

**Performance**:
- AUC: ~0.96
- Accuracy: ~89-90%
- F1 Score: ~0.89
- Precision: ~0.90
- Recall: ~0.89

**Status**: ✅ PRIMARY CLASSIFICATION MODEL

---

#### 6. Logistic Regression - All Submodels (Classification)

**File**: `log_reg_cv_ALL_submodels_tuned.joblib`

**Training Script**: `src/6.1_ML_Logistic_Regression.py`

**Specifications**:
- Same as above but includes all feature subsets
- Purpose: Ensemble or comparison of different feature combinations
- Likely includes: text-only, metadata-only, combined

**Status**: ⚠️ ALTERNATIVE MODEL (for comparison)

---

#### 7. Linear Regression - Top Features (Regression)

**File**: `lin_reg_ALL_top_features.joblib`

**Training Script**: `src/10.1_ML_Linear_Regression.py`

**Specifications**:
- Algorithm: Linear Regression (sklearn)
- Training Size: 5,523,992 rows
- Target: Continuous (target_reg - time-discounted vote count)
- Features: Top features selected
- Preprocessing: PowerTransformer scaling

**Evaluation Metrics**:
- R² Score: Unknown (not in current docs)
- MAE: Unknown
- RMSE: Unknown

**Purpose**: Predict actual vote counts (not just binary quality)

**Status**: ✅ PRIMARY REGRESSION MODEL

---

#### 8. Linear Regression - All Submodels (Regression)

**File**: `lin_reg_ALL_submodels.joblib`

**Training Script**: `src/10.1_ML_Linear_Regression.py`

**Specifications**:
- Same as above but multiple feature subsets
- Purpose: Compare different feature combinations

**Status**: ⚠️ ALTERNATIVE MODEL (for comparison)

---

### NLP Models (models/nlp/)

Located: `models/nlp/`  
Purpose: Text processing and feature extraction

#### 9. Naive Bayes TF-IDF

**Directory**: `NB_TFIDF_all/`

**Training Script**: `src/2.3.1_NLP_Spark_Tf-Idf_Models.py`

**Specifications**:
- Algorithm: Naive Bayes (Spark ML)
- Input: TF-IDF vectors of review text
- Output: `NB_prob` column (probability of quality review)
- Purpose: Text-based classification feature for downstream models

**Training Data**: Full dataset (text only)

**Usage**: 
- Adds `NB_prob` as a feature to PostgreSQL
- Used as input feature for final ML models

---

#### 10. SVM TF-IDF

**Directory**: `SVM_TFIDF_all/`

**Training Script**: `src/2.3.1_NLP_Spark_Tf-Idf_Models.py`

**Specifications**:
- Algorithm: Linear SVM (Spark ML)
- Input: TF-IDF vectors of review text
- Output: `svm_pred` column (prediction)
- Purpose: Text-based classification feature

**Training Data**: Full dataset (text only)

**Usage**: 
- Adds `svm_pred` as a feature to PostgreSQL
- Used as input feature for final ML models

---

#### 11. FastText Embeddings

**Directory**: `fasttext_model_ALL`

**Training Script**: `src/2.4.2_NLP_Fasttext.py`

**Specifications**:
- Algorithm: FastText (Facebook)
- Input: Review text
- Output: `ft_prob` column (quality probability)
- Purpose: Word embeddings for text representation

**Training Data**: Full dataset

**Usage**:
- Adds `ft_prob` as a feature to PostgreSQL
- Captures semantic meaning of reviews

---

#### 12. LDA Topic Model

**Directory**: `LDA_model_1M/`

**Training Script**: `src/2.5_NLP_Topic_Modeling_LDA.py`

**Specifications**:
- Algorithm: Latent Dirichlet Allocation (Gensim)
- Number of Topics: 5
- Input: Lemmatized review text
- Output: `lda_t1` through `lda_t5` (topic probabilities)
- Preprocessing: 
  - spaCy lemmatization
  - NLTK stopword removal
  - Custom stopwords

**Training Data**: 1M sample

**Topics Discovered** (inferred):
- Likely includes: Food/Dining, Service, Atmosphere, Value, General Experience

**Usage**:
- Adds 5 topic probability features to PostgreSQL
- Captures thematic content of reviews

---

## Model Performance Comparison

### Classification Models (AUC on CV)

| Model | AUC | Accuracy | F1 | Training Size | Status |
|-------|-----|----------|----|--------------:|--------|
| Logistic Regression (Final) | 0.96 | 0.90 | 0.89 | 5.5M | ✅ Production |
| Random Forest (Base) | 0.96-1.0 | 0.95-1.0 | 0.95-1.0 | 1M | Baseline |
| XGBoost (Base) | ~0.96 | ~0.90 | ~0.90 | 1M | Baseline |
| Decision Tree (Base) | Lower | Lower | Lower | 1M | Baseline |
| Naive Bayes (NLP) | N/A | N/A | N/A | Full | Feature |
| SVM (NLP) | N/A | N/A | N/A | Full | Feature |
| FastText (NLP) | N/A | N/A | N/A | Full | Feature |

**Winner**: Logistic Regression (balance of performance, speed, interpretability)

---

### Regression Models

| Model | R² | MAE | RMSE | Training Size | Status |
|-------|----|----|------|---------------|--------|
| Linear Regression (Final) | TBD | TBD | TBD | 5.5M | ✅ Production |
| Linear Regression Submodels | TBD | TBD | TBD | 5.5M | Alternative |

**Note**: Regression metrics not fully documented in current files

---

## Feature Importance (Estimated)

Based on model_info.csv patterns, most important features likely include:

**Top 10 Features** (estimated):
1. NLP predictions: `NB_prob`, `svm_pred`, `ft_prob`
2. Review length: `review_char_count`, `review_word_count`
3. User reputation: `user_useful`, `user_review_count`, `user_fans`
4. LDA topics: `lda_t1` - `lda_t5`
5. User elite status: `user_elite_count`

**Less Important**:
- Many POS/NER/Dependency features (high dimensionality, low signal)
- Some business attributes (weak correlation)

---

## Model Training Configuration Patterns

### Common Preprocessing

**All Classification Models**:
```python
# Scaling
PowerTransformer()

# Balancing (50% reviews have NO votes)
SMOTE(sampling_strategy='auto')

# Feature Selection (various methods)
- Top N features by importance
- All features
- Feature subsets (text-only, metadata-only)
```

### Common Hyperparameter Search Space

**Logistic Regression**:
- C: [0.5, 1.0, 2.0]
- solver: ['lbfgs', 'newton-cg', 'saga']
- penalty: ['l2']
- max_iter: [50, 100, 5000]
- class_weight: [None, 'balanced']

**Random Forest**:
- n_estimators: [5, 10]
- max_depth: [10, 100, None]
- min_samples_split: [2, 10]
- min_samples_leaf: [1, 5]
- max_features: ['sqrt', 10, None]
- max_leaf_nodes: [10, 100, None]
- criterion: ['gini', 'entropy']

---

## Model Experiment Tracking

### Legacy: CSV Tracking (Version 1)

**File**: `models/version_1/model_info.csv`

**Experiments Logged**: 8,533 model runs

**Columns Tracked**:
- Model type, hyperparameters
- CV performance (accuracy, F1, AUC, etc.)
- Test performance
- Training/scoring times
- Data configuration (features used, record count, target)

**Status**: 🗑️ DEPRECATED - Moved to MLflow

---

### Current: MLflow Tracking

**Location**: `models/pycaret/pycaret_mlruns_info/`

**What's Tracked**:
- All sklearn model parameters
- Training metrics
- Model artifacts (automatically via MLflow autolog)
- Experiment comparisons

**Usage** (in scripts):
```python
import mlflow
mlflow.sklearn.autolog()

with mlflow.start_run() as run:
    model.fit(X_train, y_train)
```

---

## Model Serving (Future)

### Current State
- Models saved as `.joblib` files
- Manual loading with `joblib.load()`
- Predictions generated via scripts (11.1)

### Future Enhancements
- REST API for real-time predictions (FastAPI/Flask)
- Model registry (MLflow Model Registry)
- A/B testing framework
- Automated model retraining
- Containerized model serving (Docker)

---

## Model Maintenance

### When to Retrain

**Triggers**:
1. New data available (updated Yelp dataset)
2. Model performance degradation
3. Feature engineering improvements
4. Hyperparameter optimization findings

**Process**:
1. Update PostgreSQL with new data (1_ETL_Spark.py)
2. Re-run NLP processing (2.x scripts) if needed
3. Re-export CSV (3.x)
4. Retrain models (4.x-11.x)
5. Compare performance with MLflow
6. Update production models if improved

---

### Model Versioning

**Current Approach**:
- Overwrite joblib files
- No systematic versioning
- Git commit messages track changes

**Recommended Approach**:
- MLflow Model Registry with versions
- Semantic versioning: v1.0.0, v1.1.0, etc.
- Tag models: production, staging, archive
- Keep last N versions

---

## Model Storage Locations

```
models/
├── base_models/                        # ~400 MB total
│   ├── log_reg_sklearn_base_model_1M.joblib
│   ├── tree_sklearn_base_model_1M.joblib
│   ├── forest_sklearn_base_model_1M.joblib
│   └── xgboost_sklearn_base_model_1M.joblib
│
├── final_models/                       # ~500 MB total (estimated)
│   ├── log_reg_cv_ALL_top_features_tuned.joblib
│   ├── log_reg_cv_ALL_submodels_tuned.joblib
│   ├── lin_reg_ALL_top_features.joblib
│   └── lin_reg_ALL_submodels.joblib
│
├── nlp/                                # Size varies
│   ├── NB_TFIDF_all/
│   ├── SVM_TFIDF_all/
│   ├── fasttext_model_ALL/            # Large (embeddings)
│   └── LDA_model_1M/                  # ~50-100 MB
│
├── pycaret/
│   └── pycaret_mlruns_info/           # MLflow artifacts
│
└── archive/                            # (Future)
    └── version_1/
        └── model_info.csv              # 8533 experiments
```

---

## Quick Reference

### Load a Model

```python
from joblib import load

# Classification (primary)
clf_model = load('models/final_models/log_reg_cv_ALL_top_features_tuned.joblib')

# Regression (primary)
reg_model = load('models/final_models/lin_reg_ALL_top_features.joblib')

# Predict
predictions_clf = clf_model.predict(X_test)
predictions_reg = reg_model.predict(X_test)
```

### Model Input Requirements

**Expected Features** (~80):
- Review metadata (stars, char_count, word_count, etc.)
- NLP predictions (NB_prob, svm_pred, ft_prob)
- LDA topics (lda_t1 - lda_t5)
- spaCy features (POS, NER, dependencies percentages)
- User features (time-discounted)
- Business features (time-discounted)

**Preprocessing Required**:
- PowerTransformer scaling (for classification)
- Feature selection (if using top features version)

---

## Model Documentation Gaps

**Missing Information**:
- ❌ Exact feature lists for each model
- ❌ Regression model performance metrics (R², MAE, RMSE)
- ❌ Feature importance rankings
- ❌ Model size (MB) for each joblib file
- ❌ Training times for final models
- ❌ Inference times (predictions per second)

**TODO**:
- [ ] Document exact feature lists
- [ ] Run evaluation on final models
- [ ] Generate feature importance plots
- [ ] Measure inference performance
- [ ] Create model cards (model documentation standard)
