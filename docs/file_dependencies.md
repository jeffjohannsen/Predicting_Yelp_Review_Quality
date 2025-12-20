# File Dependencies Map

> **Generated**: December 19, 2025  
> **Purpose**: Document which scripts depend on which data files and their execution order

---

## Pipeline Execution Order

### 1. ETL Stage - Data Loading (1.x)

**Script**: `src/1_ETL_Spark.py`

**Input Dependencies**:
- `data/full_data/original_json/yelp_academic_dataset_checkin.json`
- `data/full_data/original_json/yelp_academic_dataset_user.json`
- `data/full_data/original_json/yelp_academic_dataset_business.json`
- `data/full_data/original_json/yelp_academic_dataset_review.json`
- PostgreSQL JDBC driver: `/home/ubuntu/postgresql-42.2.20.jar` (EC2)

**Output Dependencies**:
- PostgreSQL RDS tables:
  - `text_data_train` (review text + metadata)
  - `text_data_test`
  - `non_text_data_train` (user + business features)
  - `non_text_data_test`
  - `holdout_data`

**Next Steps**: Writes to PostgreSQL → consumed by NLP scripts

---

### 2. NLP Stage - Text Processing (2.x)

#### 2.1 Basic Text Features

**Script**: `src/2.1_NLP_Basic_Text_Processing_Spark.py`

**Input Dependencies**:
- PostgreSQL tables: `text_data_train`, `text_data_test`
- PostgreSQL JDBC driver: `/home/ubuntu/postgresql-42.2.20.jar`

**Output Dependencies**:
- PostgreSQL tables (updated with new columns):
  - Adds: `review_token_count`, readability/sentiment features

**Next Steps**: Updates PostgreSQL tables → consumed by other NLP scripts

#### 2.2 Linguistic Features (spaCy)

**Script**: `src/2.2_NLP_Spacy_POS_ENT_DEP.py`

**Input Dependencies**:
- PostgreSQL tables: `text_data_train`, `text_data_test`
- spaCy model: `en_core_web_sm` (must be downloaded)

**Output Dependencies**:
- PostgreSQL tables (updated with new columns):
  - Adds: `review_perc_noun`, `review_perc_verb`, etc. (POS tags)
  - Adds: `review_perc_person`, `review_perc_org`, etc. (NER)
  - Adds: `review_perc_nsubj`, `review_perc_dobj`, etc. (dependencies)

**Next Steps**: Updates PostgreSQL tables → consumed by ML scripts

#### 2.3 TF-IDF + NB/SVM Models

**Script**: `src/2.3.1_NLP_Spark_Tf-Idf_Models.py`

**Input Dependencies**:
- PostgreSQL tables: `text_data_train`, `text_data_test`
- PostgreSQL JDBC driver: `/home/jovyan/postgresql-42.2.20.jar`

**Output Dependencies**:
- Trained models (saved to filesystem - path TBD)
- PostgreSQL tables (updated):
  - Adds: `NB_prob`, `svm_pred` (predictions from NLP models)

**Next Steps**: Model predictions added to PostgreSQL → used as features

#### 2.4 Word Embeddings

**Script 1**: `src/2.4.1_NLP_Spark_Text_Embeddings.py` (Spark NLP)

**Input Dependencies**:
- PostgreSQL tables: `text_data_train`, `text_data_test`
- PostgreSQL JDBC driver: `/home/ubuntu/postgresql-42.2.22.jar`

**Output Dependencies**:
- PostgreSQL tables (updated with embeddings)

**Script 2**: `src/2.4.2_NLP_Fasttext.py`

**Input Dependencies**:
- PostgreSQL tables (reads review_text)

**Output Dependencies**:
- FastText formatted CSV files (location TBD)
- Trained FastText model: `models/nlp/fasttext_model_ALL`
- PostgreSQL tables (updated):
  - Adds: `ft_prob` (FastText predictions)

**Next Steps**: Embedding features added to database

#### 2.5 Topic Modeling (LDA)

**Script**: `src/2.5_NLP_Topic_Modeling_LDA.py`

**Input Dependencies**:
- PostgreSQL tables: `text_data_train`, `text_data_test`
- spaCy model: `en_core_web_sm`
- NLTK data: wordnet, stopwords (downloads automatically)

**Output Dependencies**:
- Preprocessed data pickles: 
  - `data/full_data/processed_train_lda.pkl`
  - `data/full_data/processed_test_lda.pkl`
- Trained LDA model: `models/nlp/LDA_model_1M/`
- PostgreSQL tables (updated):
  - Adds: `lda_t1`, `lda_t2`, `lda_t3`, `lda_t4`, `lda_t5` (topic probabilities)

**Next Steps**: Topic features added to database

---

### 3. Data Preparation Stage (3.x)

#### 3.0 Combine Text and Non-Text Data

**Script**: `src/3_ETL_Combine_Processed_Text_Data.sql` (SQL query)

**Input Dependencies**:
- PostgreSQL tables:
  - `text_data_train` (with all NLP features added)
  - `text_data_test`
  - `non_text_data_train` (user + business features)
  - `non_text_data_test`

**Output Dependencies**:
- Combined PostgreSQL tables (or export to CSV)

**Next Steps**: Data exported to CSV for ML training

#### 3.1 Export to CSV (Notebook)

**Notebook**: `notebooks/3.1_ETL_Combined_Data_to_CSV.ipynb`

**Input Dependencies**:
- Combined PostgreSQL tables

**Output Dependencies**:
- `data/full_data/model_ready/train.csv` (~5.5M rows)
- `data/full_data/model_ready/test.csv` (~1.4M rows)

**Next Steps**: CSV files consumed by all ML scripts

---

### 4. Machine Learning Stage (4.x - 11.x)

#### 4.1 Base Models

**Script**: `src/4.1_ML_Base_Models.py`

**Input Dependencies**:
- `data/full_data/model_ready/train.csv`
- `data/full_data/model_ready/test.csv`

**Output Dependencies**:
- `models/base_models/log_reg_sklearn_base_model_1M.joblib`
- `models/base_models/tree_sklearn_base_model_1M.joblib`
- `models/base_models/forest_sklearn_base_model_1M.joblib`
- `models/base_models/xgboost_sklearn_base_model_1M.joblib`

**Next Steps**: Base models serve as benchmarks

#### 5.0 PCA Dimensionality Reduction

**Script**: `src/5_PCA_Dimensionality_Reduction.py`

**Input Dependencies**:
- `data/full_data/model_ready/train.csv`
- `data/full_data/model_ready/test.csv`

**Output Dependencies**:
- `data/full_data/model_ready/train_pca.csv` (with PCA features)
- `data/full_data/model_ready/test_pca.csv`

**Next Steps**: PCA versions available for experiments

#### 6.1 Logistic Regression (Tuned)

**Script**: `src/6.1_ML_Logistic_Regression.py`

**Input Dependencies**:
- `data/full_data/model_ready/train.csv`
- `data/full_data/model_ready/test.csv`

**Output Dependencies**:
- `models/final_models/log_reg_cv_ALL_top_features_tuned.joblib`
- `models/final_models/log_reg_cv_ALL_submodels_tuned.joblib`
- MLflow experiment tracking data

**Next Steps**: Best classification model for predictions

#### 10.1 Linear Regression

**Script**: `src/10.1_ML_Linear_Regression.py`

**Input Dependencies**:
- `data/full_data/model_ready/train.csv`
- `data/full_data/model_ready/test.csv`

**Output Dependencies**:
- `models/final_models/lin_reg_ALL_top_features.joblib`
- `models/final_models/lin_reg_ALL_submodels.joblib`
- MLflow experiment tracking data

**Next Steps**: Best regression model for predictions

#### 11.1 Add Predictions to Dataset

**Script**: `src/11.1_ETL_Add_Predictions.py`

**Input Dependencies**:
- `data/full_data/model_ready/train.csv`
- `data/full_data/model_ready/test.csv`
- `models/final_models/log_reg_cv_ALL_top_features_tuned.joblib`
- `models/final_models/lin_reg_ALL_top_features.joblib`

**Output Dependencies**:
- `data/full_data/final_predict/train_full.csv` (with predictions)
- `data/full_data/final_predict/test_full.csv`

**Next Steps**: Final datasets with predictions for ranking

---

### 5. Evaluation Stage (12.x)

#### 12.1 Review Ranking (Notebook)

**Notebook**: `notebooks/12.1_Review_Ranking.ipynb`

**Input Dependencies**:
- `data/full_data/final_predict/train_full.csv`
- `data/full_data/final_predict/test_full.csv`

**Output Dependencies**:
- `data/full_data/final_predict/train_rankings.csv`
- `data/full_data/final_predict/test_rankings.csv`
- Evaluation metrics and comparisons

**Next Steps**: Final review quality rankings

---

## Critical Path Summary

```
JSON Files (5 files, 10GB)
    ↓
1_ETL_Spark.py → PostgreSQL RDS
    ↓
2.1_NLP_Basic_Text_Processing_Spark.py → Updates PostgreSQL
    ↓
2.2_NLP_Spacy_POS_ENT_DEP.py → Updates PostgreSQL
    ↓
2.3.1_NLP_Spark_Tf-Idf_Models.py → Updates PostgreSQL + saves models
    ↓
2.4.1_NLP_Spark_Text_Embeddings.py → Updates PostgreSQL
2.4.2_NLP_Fasttext.py → Saves model + Updates PostgreSQL
    ↓
2.5_NLP_Topic_Modeling_LDA.py → Saves model + Updates PostgreSQL
    ↓
3_ETL_Combine_Processed_Text_Data.sql → Combines tables
3.1_ETL_Combined_Data_to_CSV.ipynb → Exports to CSV
    ↓
data/full_data/model_ready/train.csv, test.csv
    ↓
4.1_ML_Base_Models.py → Saves base models
6.1_ML_Logistic_Regression.py → Saves final classification model
10.1_ML_Linear_Regression.py → Saves final regression model
    ↓
11.1_ETL_Add_Predictions.py → Loads models, adds predictions
    ↓
data/full_data/final_predict/train_full.csv, test_full.csv
    ↓
12.1_Review_Ranking.ipynb → Final rankings
```

---

## Parallel Processing Opportunities

These scripts can run in parallel (after their dependencies are met):

**After PostgreSQL is populated**:
- 2.1, 2.2 can run in parallel
- After 2.1, 2.2: 2.3, 2.4, 2.5 can run in parallel

**After CSV files are created**:
- 4.1, 5.0, 6.1, 10.1 can all run in parallel (different experiments)

---

## External Dependencies

### Required Downloads/Installations:
1. **spaCy model**: `python -m spacy download en_core_web_sm`
2. **NLTK data**: Downloads automatically in scripts (wordnet, stopwords)
3. **PostgreSQL JDBC driver**: Must be in path for Spark
4. **AWS RDS credentials**: In `src/confidential.py` (not in repo)

### Environment-Specific Paths:
- **EC2**: `/home/ubuntu/`
- **Local (Jeff)**: `/home/jeff/Documents/Data_Science_Projects/Yelp_Reviews/`
- **Jupyter**: `/home/jovyan/`

---

## Notes

- Most scripts update PostgreSQL tables incrementally (add columns)
- CSV export is a one-time step to freeze data for ML experiments
- Model training scripts (4.x - 11.x) are independent and can be re-run
- Pipeline must run in order: ETL → NLP → Export → ML → Evaluation
