# Data Schemas Documentation

> **Generated**: December 19, 2025  
> **Purpose**: Document all data structures (JSON, PostgreSQL, CSV) used in the pipeline

---

## Overview

The Yelp Review Quality Prediction pipeline processes data through multiple formats:

1. **Raw Input**: 5 JSON files (~10GB) from Yelp Open Dataset
2. **Intermediate Storage**: PostgreSQL database (AWS RDS)
3. **Model-Ready Output**: CSV files (train.csv, test.csv)
4. **Final Output**: CSV with predictions

---

## 1. Raw JSON Files (Input)

**Location**: `data/full_data/original_json/`  
**Total Size**: ~10 GB  
**Format**: Newline-delimited JSON (each line is a separate JSON object)

### 1.1 yelp_academic_dataset_review.json

**Records**: ~8,000,000  
**Size**: ~6 GB

**Schema**:
```json
{
  "review_id": "string",           // Unique review identifier
  "user_id": "string",             // Foreign key to user
  "business_id": "string",         // Foreign key to business
  "stars": integer,                // Rating (1-5)
  "useful": integer,               // Count of useful votes
  "funny": integer,                // Count of funny votes
  "cool": integer,                 // Count of cool votes
  "text": "string",                // Full review text (variable length)
  "date": "YYYY-MM-DD HH:MM:SS"   // Review timestamp
}
```

**Example**:
```json
{
  "review_id": "xQY8N_XvtGbearJ5X4QryQ",
  "user_id": "OwjRMXRC0KyPrIlcjaXeFQ",
  "business_id": "-MhfebM0QIsKt87iDN-FNw",
  "stars": 2,
  "useful": 5,
  "funny": 0,
  "cool": 0,
  "text": "As someone who has worked with...",
  "date": "2015-04-15 05:21:16"
}
```

**Notes**:
- `useful + funny + cool` = total votes (target variable)
- ~50% of reviews have ZERO votes

---

### 1.2 yelp_academic_dataset_user.json

**Records**: ~2,000,000  
**Size**: ~3 GB

**Schema**:
```json
{
  "user_id": "string",             // Unique user identifier
  "name": "string",                // User display name
  "review_count": integer,         // Total reviews written
  "yelping_since": "YYYY-MM-DD",  // Account creation date
  "useful": integer,               // Total useful votes received
  "funny": integer,                // Total funny votes received
  "cool": integer,                 // Total cool votes received
  "elite": "string",               // Comma-separated years (e.g., "2012,2013")
  "friends": "string",             // Comma-separated user_ids
  "fans": integer,                 // Number of fans
  "average_stars": float,          // Average rating given
  "compliment_hot": integer,       // Compliment counts...
  "compliment_more": integer,
  "compliment_profile": integer,
  "compliment_cute": integer,
  "compliment_list": integer,
  "compliment_note": integer,
  "compliment_plain": integer,
  "compliment_cool": integer,
  "compliment_funny": integer,
  "compliment_writer": integer,
  "compliment_photos": integer
}
```

**Example**:
```json
{
  "user_id": "ntlvfPzc8eglqvk92iDIAw",
  "name": "Rafael",
  "review_count": 553,
  "yelping_since": "2007-07-06",
  "useful": 628,
  "funny": 225,
  "cool": 227,
  "elite": "2012,2013,2014,2015,2016,2017",
  "friends": "user1,user2,user3,...",
  "fans": 98,
  "average_stars": 3.69,
  "compliment_hot": 35,
  ...
}
```

**Notes**:
- `friends` field can be very long (thousands of user_ids)
- Elite status is year-based, not boolean

---

### 1.3 yelp_academic_dataset_business.json

**Records**: ~210,000  
**Size**: ~150 MB

**Schema**:
```json
{
  "business_id": "string",         // Unique business identifier
  "name": "string",                // Business name
  "address": "string",             // Street address
  "city": "string",
  "state": "string",               // 2-letter state code
  "postal_code": "string",
  "latitude": float,               // GPS coordinates
  "longitude": float,
  "stars": float,                  // Average business rating
  "review_count": integer,         // Total reviews
  "is_open": integer,              // 1=open, 0=closed
  "attributes": {                  // Nested object (varies by business)
    "RestaurantsPriceRange2": "string",
    "BusinessParking": {...},
    "BikeParking": "boolean",
    ...
  },
  "categories": "string",          // Comma-separated categories
  "hours": {                       // Operating hours (nested)
    "Monday": "0:00-0:00",
    ...
  }
}
```

**Example**:
```json
{
  "business_id": "f9NumwFMBDn751xgFiRbNA",
  "name": "The Range At Lake Norman",
  "address": "10913 Bailey Rd",
  "city": "Cornelius",
  "state": "NC",
  "postal_code": "28031",
  "latitude": 35.4627242,
  "longitude": -80.8526119,
  "stars": 3.5,
  "review_count": 36,
  "is_open": 1,
  "attributes": {...},
  "categories": "Active Life, Gun/Rifle Ranges, Guns & Ammo, Shopping",
  "hours": {...}
}
```

**Notes**:
- Attributes are nested and vary widely by business type
- Currently only basic fields are used (lat/lon, stars, review_count)

---

### 1.4 yelp_academic_dataset_checkin.json

**Records**: ~175,000  
**Size**: ~500 MB

**Schema**:
```json
{
  "business_id": "string",         // Foreign key to business
  "date": "string"                 // Comma-separated timestamps
}
```

**Example**:
```json
{
  "business_id": "--1UhMGODdWsrMastO9DZw",
  "date": "2016-04-26 19:49:16, 2016-08-30 18:36:57, 2016-10-15 02:45:18, ..."
}
```

**Notes**:
- `date` field is a SINGLE string with comma-separated timestamps
- Used to derive `num_checkins`, `checkin_min`, `checkin_max`

---

### 1.5 yelp_academic_dataset_tip.json

**Records**: ~1,300,000  
**Size**: ~200 MB

**Schema**:
```json
{
  "user_id": "string",             // Foreign key to user
  "business_id": "string",         // Foreign key to business
  "text": "string",                // Tip text (shorter than reviews)
  "date": "YYYY-MM-DD HH:MM:SS",  // Timestamp
  "compliment_count": integer      // Compliments received
}
```

**Status**: ❌ NOT CURRENTLY USED in pipeline

---

## 2. PostgreSQL Database Schema (Intermediate)

**Location**: AWS RDS PostgreSQL instance  
**Connection Details**: Stored in `src/confidential.py` (not in repo)  
**Database Name**: `yelp_db` (assumed)

### 2.1 Core Tables (from ETL)

Created by: `src/1_ETL_Spark.py`

#### Table: `all_data_train`

**Records**: ~5,523,992 (80% of 80% of total)  
**Purpose**: Combined training data with user/business/review info

**Schema**:
```sql
CREATE TABLE all_data_train (
    review_id VARCHAR(22) PRIMARY KEY,
    user_id VARCHAR(22) NOT NULL,
    business_id VARCHAR(22) NOT NULL,
    
    -- Business features
    biz_latitude FLOAT,
    biz_longitude FLOAT,
    biz_postal_code VARCHAR(10),
    biz_state VARCHAR(2),
    biz_avg_stars FLOAT,              -- Business average rating
    biz_review_count INTEGER,         -- Total reviews for business
    biz_checkin_count INTEGER,        -- Total checkins
    biz_min_checkin_date TIMESTAMP,
    biz_max_checkin_date TIMESTAMP,
    
    -- User features
    user_yelping_since TIMESTAMP,
    user_elite_count INTEGER,         -- Number of elite years
    user_elite_min INTEGER,           -- First elite year
    user_elite_max INTEGER,           -- Last elite year
    user_avg_stars FLOAT,             -- User's average rating
    user_review_count INTEGER,        -- User's total reviews
    user_fan_count INTEGER,
    user_friend_count INTEGER,        -- Count from friends string
    user_compliment_count INTEGER,    -- Sum of all compliment types
    user_ufc_count INTEGER,           -- User's total votes (useful+funny+cool)
    
    -- Review features
    review_date TIMESTAMP,
    review_stars INTEGER,             -- Rating (1-5)
    review_text TEXT,                 -- Full review text
    
    -- Target variables
    target_ufc_count INTEGER,         -- useful + funny + cool votes
    target_ufc_bool VARCHAR(5)        -- "True" if > 0, else "False"
);
```

**Indexes**: Primary key on `review_id`

---

#### Table: `all_data_test`

**Records**: ~1,380,998 (20% of 80% of total)  
**Purpose**: Test data (same schema as train)

---

#### Table: `holdout_data`

**Records**: ~1,726,248 (20% of total)  
**Purpose**: Holdout set for final evaluation (same schema)

---

### 2.2 Text Processing Tables

Created by: `src/2.x_NLP_*.py` scripts

#### Table: `text_data_train`

**Records**: 5,523,992  
**Purpose**: Review text with targets only

**Schema**:
```sql
CREATE TABLE text_data_train (
    review_id VARCHAR(22) PRIMARY KEY,
    review_stars INTEGER,
    review_text TEXT,
    target_ufc_bool VARCHAR(5),
    target_ufc_count INTEGER
);
```

---

#### Table: `text_basic_train`

**Records**: 5,523,992  
**Created by**: `src/2.1_NLP_Basic_Text_Processing_Spark.py`

**Schema**:
```sql
CREATE TABLE text_basic_train (
    review_id VARCHAR(22) PRIMARY KEY,
    grade_level FLOAT,              -- Readability (Flesch-Kincaid)
    polarity FLOAT,                 -- Sentiment (-1 to 1)
    subjectivity FLOAT,             -- Subjectivity (0 to 1)
    word_count INTEGER,
    character_count INTEGER,
    num_count INTEGER,              -- Number of digits
    uppercase_count INTEGER,
    "#_@_count" INTEGER,            -- Special chars
    sentence_count INTEGER,
    lexicon_count INTEGER,          -- Unique words
    syllable_count INTEGER,
    avg_word_length FLOAT
);
```

---

#### Table: `text_spacy_train`

**Records**: 5,523,992  
**Created by**: `src/2.2_NLP_Spacy_POS_ENT_DEP.py`

**Schema** (170+ columns):
```sql
CREATE TABLE text_spacy_train (
    review_id VARCHAR(22) PRIMARY KEY,
    
    -- Token features
    token_count INTEGER,
    stopword_count INTEGER,
    stopword_perc FLOAT,
    ent_count INTEGER,
    ent_perc FLOAT,
    
    -- Part-of-Speech (POS) tags (17 types × 2 = 34 columns)
    pos_adj_perc FLOAT,
    pos_adj_count INTEGER,
    pos_adp_perc FLOAT,
    pos_adp_count INTEGER,
    pos_adv_perc FLOAT,
    pos_adv_count INTEGER,
    pos_aux_perc FLOAT,
    pos_aux_count INTEGER,
    pos_conj_perc FLOAT,
    pos_conj_count INTEGER,
    pos_det_perc FLOAT,
    pos_det_count INTEGER,
    pos_intj_perc FLOAT,
    pos_intj_count INTEGER,
    pos_noun_perc FLOAT,
    pos_noun_count INTEGER,
    pos_num_perc FLOAT,
    pos_num_count INTEGER,
    pos_part_perc FLOAT,
    pos_part_count INTEGER,
    pos_pron_perc FLOAT,
    pos_pron_count INTEGER,
    pos_propn_perc FLOAT,
    pos_propn_count INTEGER,
    pos_punct_perc FLOAT,
    pos_punct_count INTEGER,
    pos_sconj_perc FLOAT,
    pos_sconj_count INTEGER,
    pos_sym_perc FLOAT,
    pos_sym_count INTEGER,
    pos_verb_perc FLOAT,
    pos_verb_count INTEGER,
    pos_x_perc FLOAT,
    pos_x_count INTEGER,
    
    -- Dependency tags (45 types × 2 = 90 columns)
    dep_root_perc FLOAT,
    dep_root_count INTEGER,
    dep_acl_perc FLOAT,
    dep_acl_count INTEGER,
    -- ... 43 more dep types
    dep_xcomp_perc FLOAT,
    dep_xcomp_count INTEGER,
    
    -- Named Entity Recognition (18 types × 2 = 36 columns)
    ent_cardinal_perc FLOAT,
    ent_cardinal_count INTEGER,
    ent_date_perc FLOAT,
    ent_date_count INTEGER,
    ent_event_perc FLOAT,
    ent_event_count INTEGER,
    ent_fac_perc FLOAT,
    ent_fac_count INTEGER,
    ent_gpe_perc FLOAT,
    ent_gpe_count INTEGER,
    ent_language_perc FLOAT,
    ent_language_count INTEGER,
    ent_law_perc FLOAT,
    ent_law_count INTEGER,
    ent_loc_perc FLOAT,
    ent_loc_count INTEGER,
    ent_money_perc FLOAT,
    ent_money_count INTEGER,
    ent_norp_perc FLOAT,
    ent_norp_count INTEGER,
    ent_ordinal_perc FLOAT,
    ent_ordinal_count INTEGER,
    ent_org_perc FLOAT,
    ent_org_count INTEGER,
    ent_percent_perc FLOAT,
    ent_percent_count INTEGER,
    ent_person_perc FLOAT,
    ent_person_count INTEGER,
    ent_product_perc FLOAT,
    ent_product_count INTEGER,
    ent_quantity_perc FLOAT,
    ent_quantity_count INTEGER,
    ent_time_perc FLOAT,
    ent_time_count INTEGER,
    ent_work_of_art_perc FLOAT,
    ent_work_of_art_count INTEGER
);
```

**Total Columns**: ~170 (token + POS + DEP + NER features)

---

#### Table: `text_nb_train`

**Records**: 5,523,992  
**Created by**: `src/2.3.1_NLP_Spark_Tf-Idf_Models.py`

**Schema**:
```sql
CREATE TABLE text_nb_train (
    review_id VARCHAR(22) PRIMARY KEY,
    NB_tfidf_true_prob FLOAT        -- Naive Bayes probability
);
```

---

#### Table: `text_svm_train`

**Records**: 5,523,992  
**Created by**: `src/2.3.1_NLP_Spark_Tf-Idf_Models.py`

**Schema**:
```sql
CREATE TABLE text_svm_train (
    review_id VARCHAR(22) PRIMARY KEY,
    svm_pred FLOAT                  -- SVM prediction
);
```

---

#### Table: `text_ft_train`

**Records**: 5,523,992  
**Created by**: `src/2.4.2_NLP_Fasttext.py`

**Schema**:
```sql
CREATE TABLE text_ft_train (
    review_id VARCHAR(22) PRIMARY KEY,
    ft_quality_prob FLOAT           -- FastText quality probability
);
```

---

#### Table: `text_lda_train`

**Records**: 5,523,992  
**Created by**: `src/2.5_NLP_Topic_Modeling_LDA.py`

**Schema**:
```sql
CREATE TABLE text_lda_train (
    review_id VARCHAR(22) PRIMARY KEY,
    topic_0_lda FLOAT,              -- Topic 1 probability
    topic_1_lda FLOAT,              -- Topic 2 probability
    topic_2_lda FLOAT,              -- Topic 3 probability
    topic_3_lda FLOAT,              -- Topic 4 probability
    topic_4_lda FLOAT               -- Topic 5 probability
);
```

---

### 2.3 Combined Text Table

**Table**: `text_combined_train`  
**Created by**: `src/3_ETL_Combine_Processed_Text_Data.sql`

**Records**: 5,523,992  
**Purpose**: Single table with ALL text features joined

**Schema**: ~200 columns
- All columns from `text_data_train` (5)
- All columns from `text_basic_train` (13)
- All columns from `text_spacy_train` (170)
- All columns from `text_nb_train` (1)
- All columns from `text_svm_train` (1)
- All columns from `text_ft_train` (1)
- All columns from `text_lda_train` (5)

**Note**: Similar tables exist for test: `text_combined_test`

---

## 3. Model-Ready CSV Files (Output)

**Location**: `data/full_data/model_ready/`  
**Format**: CSV with headers  
**Encoding**: UTF-8

### 3.1 train.csv

**Records**: 5,523,992  
**Size**: ~2-3 GB (estimated)  
**Created by**: Exporting `text_combined_train` from PostgreSQL

**Schema**: ~80 feature columns

**Column List**:
```
review_id                  (index, not used in training)
target_clf                 (binary: True/False)
target_reg                 (continuous: time-discounted vote count)
review_stars               (1-5)
nb_prob                    (Naive Bayes probability)
svm_pred                   (SVM prediction)
ft_prob                    (FastText probability)
lda_t1, lda_t2, ..., lda_t5  (LDA topic probabilities)
grade_level                (readability)
polarity                   (sentiment)
subjectivity               (sentiment)
word_cnt, character_cnt, num_cnt, uppercase_cnt, #@_cnt
sentence_cnt, lexicon_cnt, syllable_cnt, avg_word_len
token_cnt, stopword_cnt, stopword_pct
ent_cnt, ent_pct
pos_adj_pct, pos_adj_cnt, ... (17 POS types × 2)
dep_root_pct, dep_root_cnt, ... (45 DEP types × 2)
ent_cardinal_pct, ent_cardinal_cnt, ... (18 NER types × 2)
```

**Data Types** (as loaded in Python):
```python
datatypes = {
    "target_reg": "int16",
    "review_stars": "int16",
    "NB_prob": "float32",
    "svm_pred": "float32",
    "ft_prob": "float32",
    "lda_t1": "float32",
    "lda_t2": "float32",
    "lda_t3": "float32",
    "lda_t4": "float32",
    "lda_t5": "float32",
    "grade_level": "float32",
    "polarity": "float32",
    "subjectivity": "float32",
    "word_cnt": "int16",
    "character_cnt": "int16",
    "num_cnt": "int16",
    "uppercase_cnt": "int16",
    "#@_cnt": "int16",
    "sentence_cnt": "int16",
    "lexicon_cnt": "int16",
    "syllable_cnt": "int16",
    "avg_word_len": "float32",
    "token_cnt": "int16",
    "stopword_cnt": "int16",
    "stopword_pct": "float32",
    "ent_cnt": "int16",
    "ent_pct": "float32",
    # ... all POS/DEP/NER features as float32 or int16
}
```

**Memory Optimization**: Explicit dtypes reduce memory by ~50%

---

### 3.2 test.csv

**Records**: 1,380,998  
**Size**: ~500 MB - 1 GB (estimated)  
**Schema**: Same as train.csv

---

### 3.3 holdout.csv

**Records**: 1,726,248  
**Status**: ⚠️ May or may not exist (not mentioned in scripts)

---

## 4. Final Prediction Files (Output)

**Location**: `data/full_data/final_predict/`  
**Created by**: `src/11.1_ETL_Add_Predictions.py`

### 4.1 train_with_predictions.csv

**Schema**: train.csv + prediction columns
- All original columns
- `log_reg_pred_clf` - Logistic regression classification prediction
- `log_reg_prob_clf` - Logistic regression probability
- `lin_reg_pred` - Linear regression prediction

---

### 4.2 test_with_predictions.csv

**Schema**: Same as train_with_predictions.csv

---

## 5. Data Relationships

### Entity-Relationship Overview

```
USER (2M)
    |
    | 1:N
    |
REVIEW (8M) ---- N:1 ---- BUSINESS (210K)
    |                          |
    | 1:1                      | 1:1
    |                          |
TEXT FEATURES           CHECKIN (175K)
(NLP processing)
```

### Foreign Keys
- `review.user_id` → `user.user_id`
- `review.business_id` → `business.business_id`
- `checkin.business_id` → `business.business_id`

---

## 6. Data Splits

### Random Split Strategy

**Seed**: 12345 (for reproducibility)

```
All Data (8,631,238 reviews)
    |
    +-- 80% Working Data (6,904,990)
    |       |
    |       +-- 80% Train (5,523,992)
    |       |
    |       +-- 20% Test (1,380,998)
    |
    +-- 20% Holdout (1,726,248)
```

**Purpose**:
- Train: Model training
- Test: Hyperparameter tuning, model selection
- Holdout: Final unbiased evaluation (NOT USED YET)

---

## 7. Special Considerations

### Time Discounting

**Applies to**: User and business features in train/test CSV

**Formula**:
```python
# Simplified version
value_at_review_time = (current_value / days_since_start) * days_from_start_to_review
```

**Affected Fields**:
- User: review_count, useful, funny, cool, fans, friends
- Business: review_count, checkin_count

**NOT Discounted**:
- Review text features (inherent to review)
- User average_stars (assumed constant)
- Business stars (assumed constant)

### Missing Values

**Strategy**:
- Left joins in SQL allow NULLs
- Business without checkins → NULL checkin fields
- Users without elite status → 0 for elite fields

---

## 8. Data Quality Notes

### Known Issues

**JSON Files**:
- Some review text contains special characters, emojis
- Elite field is string "20,20" sometimes instead of "2020"
- Friends field can be VERY long (memory issue)

**PostgreSQL**:
- Text columns are large (review_text)
- Many sparse features (POS/DEP/NER percentages often 0)

**CSV Files**:
- Large file size (multi-GB)
- Loading full CSV requires 10+ GB RAM
- Always use `nrows` for testing

---

## 9. Schema Evolution

### Version 0 (Deprecated)
- Used MongoDB for storage
- Different table structure
- Files in `src/version_0/`

### Version 1 (Current)
- PostgreSQL for relational benefits
- Spark for ETL efficiency
- Normalized text processing tables

### Version 2 (Proposed)
- Add indexes on foreign keys
- Partition large tables by date
- Compress text_spacy table (many sparse columns)
- Add data validation constraints

---

## 10. Quick Reference

### Load CSV in Python

```python
import pandas as pd

# Memory-efficient loading
datatypes = {
    "target_reg": "int16",
    "review_stars": "int16",
    # ... (see full list in section 3.1)
}

df_train = pd.read_csv(
    "data/full_data/model_ready/train.csv",
    nrows=100000,  # Sample for testing
    dtype=datatypes
)
```

### Query PostgreSQL

```python
import psycopg2
from src.confidential import db_user, db_password, db_host, db_port, db_name

conn = psycopg2.connect(
    user=db_user,
    password=db_password,
    host=db_host,
    port=db_port,
    database=db_name
)

query = "SELECT * FROM text_combined_train LIMIT 1000"
df = pd.read_sql_query(query, conn)
conn.close()
```

### Load with Spark

```python
import pyspark as ps

spark = ps.sql.SparkSession.builder.getOrCreate()

df = spark.read.csv(
    "data/full_data/model_ready/train.csv",
    header=True,
    inferSchema=True
)
```

---

## 11. Documentation Gaps

**Missing Details**:
- ❌ Exact PostgreSQL table DDL statements
- ❌ Database indexes
- ❌ Actual file sizes for CSV files
- ❌ Memory requirements for full data load
- ❌ Row counts for all intermediate tables
- ❌ Data validation rules (constraints)

**TODO**:
- [ ] Generate complete DDL from database
- [ ] Measure actual CSV file sizes
- [ ] Document data quality checks
- [ ] Create data dictionary with definitions
- [ ] Add example queries for common use cases
