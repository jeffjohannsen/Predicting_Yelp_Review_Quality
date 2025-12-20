# Data Pipeline Flow Diagram

> **Generated**: December 19, 2025  
> **Purpose**: Visual representation of the complete data pipeline

---

## High-Level Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    YELP OPEN DATASET (10GB)                     │
│  • 8M reviews  • 2M users  • 210K businesses  • 175K checkins  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: ETL & LOADING                       │
│                   (1_ETL_Spark.py)                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Read 5 JSON files with Spark                           │  │
│  │ • Transform (SQL queries, data cleaning)                 │  │
│  │ • Split: Train (80%) / Test (20%)                        │  │
│  │ • Load → PostgreSQL RDS (AWS)                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     POSTGRESQL DATABASE                         │
│  Tables: text_data_train, text_data_test,                      │
│          non_text_data_train, non_text_data_test               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 2: NLP PROCESSING                       │
│                   (2.1 - 2.5 scripts)                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 2.1 Basic Text Features (Spark)                          │  │
│  │     • Token counts, readability, sentiment               │  │
│  │     • Updates: PostgreSQL tables                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 2.2 Linguistic Features (spaCy)                          │  │
│  │     • POS tags (noun %, verb %, etc.)                    │  │
│  │     • Named Entities (person %, org %, etc.)             │  │
│  │     • Dependencies (subject, object, etc.)               │  │
│  │     • Updates: PostgreSQL tables                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 2.3 TF-IDF + NB/SVM Models (Spark)                       │  │
│  │     • TF-IDF vectorization                               │  │
│  │     • Train: Naive Bayes, SVM                            │  │
│  │     • Outputs: NB_prob, svm_pred predictions             │  │
│  │     • Saves: models/nlp/NB_TFIDF_all, SVM_TFIDF_all      │  │
│  │     • Updates: PostgreSQL tables with predictions        │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 2.4 Word Embeddings                                      │  │
│  │     • 2.4.1: Spark NLP (Word2Vec)                        │  │
│  │     • 2.4.2: FastText                                    │  │
│  │     • Outputs: ft_prob predictions                       │  │
│  │     • Saves: models/nlp/fasttext_model_ALL               │  │
│  │     • Updates: PostgreSQL tables                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 2.5 Topic Modeling (Gensim LDA)                          │  │
│  │     • Preprocessing: lemmatization, stopwords            │  │
│  │     • Train: LDA (5 topics)                              │  │
│  │     • Outputs: lda_t1 - lda_t5 probabilities             │  │
│  │     • Saves: models/nlp/LDA_model_1M                     │  │
│  │     • Updates: PostgreSQL tables                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           STAGE 3: DATA COMBINATION & EXPORT                    │
│           (3_ETL_Combine_Processed_Text_Data.sql)              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • SQL JOIN: text_data + non_text_data                    │  │
│  │ • Combines: Review text features + User features +       │  │
│  │             Business features + NLP predictions           │  │
│  │ • Export → CSV files                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL-READY DATA                           │
│  data/full_data/model_ready/train.csv (5.5M rows, 80+ features) │
│  data/full_data/model_ready/test.csv (1.4M rows, 80+ features) │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE 4: MACHINE LEARNING                       │
│                 (4.x - 11.x scripts)                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 4.1 Base Models (1M sample)                              │  │
│  │     • Logistic Regression                                │  │
│  │     • Decision Tree                                      │  │
│  │     • Random Forest                                      │  │
│  │     • XGBoost                                            │  │
│  │     • Saves: models/base_models/*.joblib                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 5.0 PCA Dimensionality Reduction (optional)              │  │
│  │     • StandardScaler + PCA                               │  │
│  │     • Outputs: train_pca.csv, test_pca.csv               │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 6.1 Logistic Regression (Tuned, Full Dataset)            │  │
│  │     • Cross-validation tuning                            │  │
│  │     • PowerTransformer + SMOTE                           │  │
│  │     • Target: T2_CLS_ufc_>0 (binary classification)      │  │
│  │     • MLflow experiment tracking                         │  │
│  │     • Saves: models/final_models/log_reg_cv_ALL_*.joblib │  │
│  │     • AUC: ~0.96, Accuracy: ~90%                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 10.1 Linear Regression (Full Dataset)                    │  │
│  │     • Regression target: target_reg (vote count)         │  │
│  │     • Feature selection                                  │  │
│  │     • Saves: models/final_models/lin_reg_ALL_*.joblib    │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 11.1 Add Predictions to Dataset                          │  │
│  │     • Load: log_reg and lin_reg models                   │  │
│  │     • Generate: classification & regression predictions  │  │
│  │     • Outputs: train_full.csv, test_full.csv             │  │
│  │     • Location: data/full_data/final_predict/            │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                STAGE 5: EVALUATION & RANKING                    │
│                (12.1_Review_Ranking.ipynb)                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Compare: Classification vs Regression predictions       │  │
│  │ • Metrics: MAE, RMSE, R² for ranking quality              │  │
│  │ • Generate: Review quality rankings                       │  │
│  │ • Outputs: train_rankings.csv, test_rankings.csv          │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ FINAL RANKINGS │
                    │ (By Business)  │
                    └────────────────┘
```

---

## Detailed Feature Flow

```
JSON Files → PostgreSQL → NLP Processing → CSV Export → ML Models → Predictions

Features Added at Each Stage:
═══════════════════════════════════════════════════════════════════════

STAGE 1 (ETL):
├── Review Features (9)
│   ├── review_id, user_id, business_id
│   ├── review_stars, review_date, review_text
│   └── review_useful, review_funny, review_cool
├── User Features (22)
│   ├── user_review_count, user_yelping_since
│   ├── user_useful, user_funny, user_cool, user_fans
│   ├── user_friend_count, user_elite_count
│   ├── user_average_stars_given
│   └── user_compliment_* (10 types)
└── Business Features (30+)
    ├── restaurant_name, restaurant_address, restaurant_city
    ├── restaurant_overall_stars, restaurant_review_count
    ├── restaurant_latitude, restaurant_longitude
    ├── restaurant_price_range, restaurant_categories
    ├── restaurant_checkin_count, checkin_min, checkin_max
    └── restaurant_is_open

STAGE 2.1 (Basic Text - Spark):
├── review_token_count
├── review_char_count
├── review_word_count
├── review_char_per_word
├── grade_level (readability)
├── polarity (sentiment)
└── subjectivity

STAGE 2.2 (Linguistic - spaCy):
├── POS Tags (17 features)
│   └── review_perc_noun, review_perc_verb, review_perc_adj, etc.
├── Named Entities (13 features)
│   └── review_perc_person, review_perc_org, review_perc_gpe, etc.
└── Dependencies (30+ features)
    └── review_perc_nsubj, review_perc_dobj, review_perc_root, etc.

STAGE 2.3 (TF-IDF Models):
├── NB_prob (Naive Bayes probability)
└── svm_pred (SVM prediction)

STAGE 2.4 (Embeddings):
└── ft_prob (FastText probability)

STAGE 2.5 (Topic Modeling):
├── lda_t1 (Topic 1 probability)
├── lda_t2 (Topic 2 probability)
├── lda_t3 (Topic 3 probability)
├── lda_t4 (Topic 4 probability)
└── lda_t5 (Topic 5 probability)

STAGE 3 (Target Creation):
├── target_reg (time-discounted vote count for regression)
└── T2_CLS_ufc_>0 (binary: has votes or not)

TOTAL: 80+ Features for ML Models
```

---

## Data Volume Flow

```
┌──────────────────┐
│   JSON Files     │  10 GB (uncompressed)
│   5 files        │  8M reviews, 2M users, 210K businesses
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  PostgreSQL RDS  │  Size: Unknown (compressed on RDS)
│  Multiple tables │  Rows: Train (~6.4M) + Test (~1.6M)
└────────┬─────────┘
         │
         │  (NLP processing adds ~60 new columns)
         │
         ▼
┌──────────────────┐
│  CSV Files       │  Size: Large (GB-scale)
│  train.csv       │  5,523,992 rows × 80+ columns
│  test.csv        │  1,382,379 rows × 80+ columns
└────────┬─────────┘
         │
         │  (ML training samples subsets)
         │
         ▼
┌──────────────────┐
│  Trained Models  │  Size: ~100-500 MB each (joblib)
│  Base: 4 models  │  Final: 4 models
│  NLP: 4 models   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Final Datasets  │
│  train_full.csv  │  5.5M rows × 85+ columns (with predictions)
│  test_full.csv   │  1.4M rows × 85+ columns
│  *_rankings.csv  │  Review rankings per business
└──────────────────┘
```

---

## Time Discounting Flow

```
Problem: Dataset from 2020, reviews from 2004-2020
         Recent reviews have less time to accumulate votes
         
┌─────────────────────────────────────────────────────────────┐
│                   TIME DISCOUNTING                          │
│                                                             │
│  Review from 2004:                                          │
│  └─ 16 years to get votes → Discount DOWN                  │
│                                                             │
│  Review from 2019:                                          │
│  └─ 1 year to get votes → Discount UP                      │
│                                                             │
│  Formula (simplified):                                      │
│  value_at_review_time =                                     │
│      (current_value / days_since_start) *                   │
│      days_from_start_to_review                              │
└─────────────────────────────────────────────────────────────┘

Applied to:
├── TARGET: review_useful + review_funny + review_cool → target_reg
├── USER: user_review_count, user_useful/funny/cool, user_fans
└── BUSINESS: restaurant_review_count, restaurant_checkin_count

NOT applied to:
├── review_stars (snapshot at time of review)
├── user_average_stars_given (assumed constant)
└── restaurant_overall_stars (assumed constant)
```

---

## Parallel Processing Opportunities

```
STAGE 1: ETL (Sequential - must complete first)
    ↓
STAGE 2: NLP Processing (Can parallelize)
    ├─ 2.1 Basic Text ──┐
    ├─ 2.2 spaCy ────────┼─→ All can run simultaneously
    ├─ 2.3 TF-IDF ───────┤   (independently update PostgreSQL)
    ├─ 2.4 Embeddings ───┤
    └─ 2.5 LDA ──────────┘
    ↓
STAGE 3: Export (Sequential - needs all NLP features)
    ↓
STAGE 4: ML Training (Can parallelize)
    ├─ 4.1 Base Models ──┐
    ├─ 5.0 PCA ──────────┼─→ All experiments independent
    ├─ 6.1 LogReg ───────┤
    └─ 10.1 LinReg ──────┘
    ↓
STAGE 5: Final Predictions & Ranking (Sequential)
```

---

## Technology Stack per Stage

| Stage | Primary Tech | Secondary Tech | Storage |
|-------|--------------|----------------|---------|
| 1. ETL | PySpark | SparkSQL | PostgreSQL RDS |
| 2.1 Text | Spark | TextBlob | PostgreSQL |
| 2.2 Linguistic | spaCy | PostgreSQL | PostgreSQL |
| 2.3 TF-IDF | Spark ML | Naive Bayes, SVM | PostgreSQL + Models |
| 2.4 Embeddings | Spark NLP, fastText | - | PostgreSQL + Models |
| 2.5 Topics | Gensim LDA | NLTK, spaCy | PostgreSQL + Models |
| 3. Export | pandas, SQL | - | CSV Files |
| 4. ML | scikit-learn | XGBoost, MLflow | Joblib models |
| 5. Ranking | pandas | - | CSV Files |

---

## Critical Dependencies

```
External Services:
├── PostgreSQL RDS (AWS) ← Must be running
├── S3 (optional) ← For data storage
└── EC2 (optional) ← For compute

External Libraries:
├── spaCy model: en_core_web_sm ← Download required
├── NLTK data: wordnet, stopwords ← Auto-downloads
└── PostgreSQL JDBC driver ← Required for Spark

Environment Files:
├── src/confidential.py ← DB credentials (not in repo)
└── .env (future) ← Local configuration
```

---

## Execution Order (Cannot Skip Steps)

1. ✅ MUST run 1_ETL_Spark.py FIRST (loads all data)
2. ✅ MUST complete ALL 2.x scripts (NLP features)
3. ✅ MUST run 3.x export (creates CSV files)
4. ✅ CAN run 4.x-11.x in any order (ML experiments)
5. ✅ MUST run 11.1 before 12.1 (add predictions → rank)

**Pipeline Breaking Points**:
- If PostgreSQL goes down → Cannot run 2.x or 3.x
- If CSV files missing → Cannot run 4.x+
- If models missing → Cannot run 11.1+
