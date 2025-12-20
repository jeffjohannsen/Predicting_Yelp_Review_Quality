# Yelp Review Quality Prediction - TODO

> **Last Updated**: December 19, 2025  
> **Current Sprint**: Pipeline Refactoring - Modernizing 2020-Era Academic Project
>
> **Project Philosophy**: This project represents a **2020-era academic ML approach** using traditional feature engineering, classical NLP (TF-IDF, LDA, fastText), and sklearn models. We are refactoring it for **code quality, structure, and reproducibility** while preserving the original methodology. A separate **SOTA rebuild** using transformers/LLMs will come later as Phase 2.
>
> **Why Two Phases?**
> 1. **Phase 1 (Current)**: Master the classical ML pipeline, improve code structure, demonstrate understanding of traditional NLP/ML
> 2. **Phase 2 (Future)**: Rebuild with SOTA 2025 tech (BERT, sentence-transformers, modern architectures) and compare approaches
>
> **Project Management Approach**: Working through the pipeline stage-by-stage, refactoring each script and consolidating notebooks as we go. Each pipeline stage is a focused mini-project.

---

## 🎯 ACTIVE SPRINT - PHASE 1

### Pipeline Refactoring - PostgreSQL to Parquet Migration (2020-Era Approach)

**Goal**: Refactor the 2020-era data pipeline to use Parquet files instead of PostgreSQL, consolidate src/notebook duplicates, and establish production-ready code structure **while keeping the original feature engineering methodology**.

**Why**: AWS RDS PostgreSQL is no longer available. Need to modernize storage (Parquet is 15-45x faster, 44% smaller), consolidate duplicate code, and prepare for reproducible execution.

**What We're Keeping** (2020-era approach):
- Time discounting methodology
- TF-IDF + Naive Bayes/SVM
- FastText embeddings  
- Gensim LDA topic modeling
- spaCy linguistic features (POS, NER, dependencies)
- Traditional ML models (LogReg, RandomForest, XGBoost)
- 80+ engineered features

**What We're Improving**:
- Code structure and modularity
- Configuration management
- Error handling and logging
- Documentation
- Storage format (PostgreSQL → Parquet)
- Reproducibility

**Strategy**: Go through pipeline stage-by-stage (ETL → NLP → ML), refactoring src scripts and consolidating/archiving notebooks at each step.

---

## 📍 PIPELINE STAGES

### Stage 0: Foundation (COMPLETED ✅)

**Completed Work**:
- [x] Phase 1 Discovery: Created 11 comprehensive documentation files
- [x] Data conversion: JSON (12 GB) → Parquet (6.7 GB, 12.3M records validated)
- [x] Infrastructure: config.py, requirements.txt, .env support
- [x] Cleanup: Deleted version_0 (MongoDB), archived version_1 critical files
- [x] Git: Consolidated to single main branch, comprehensive .gitignore
- [x] Environment: Java 17, PySpark 4.1.0, PyArrow 22.0.0 installed

**Artifacts**:
- `data/parquet_2021/` - All 5 Parquet files ready
- `src/config.py` - Centralized configuration with Parquet paths
- `docs/` - 11 documentation files
- `archive/version_1/` - Preserved time discounting and feature engineering logic

---

### Stage 1: ETL - Data Loading & Transformation

**Current State**: `1_ETL_Spark.py` (337 lines) reads JSON → writes PostgreSQL

**Goal**: Refactor to read Parquet → write processed Parquet

**Tasks**:
- [ ] **1.1**: Read `1_ETL_Spark.py` and understand current logic
- [ ] **1.2**: Extract time discounting formulas from `archive/version_1/eda_prep.py`
- [ ] **1.3**: Create `src/utils/time_discount.py` module
- [ ] **1.4**: Refactor `1_ETL_Spark.py`:
  - Read from Parquet instead of JSON
  - Implement time discounting for user/business features
  - Write to `data/processed/01_etl_output/` as Parquet
  - Remove PostgreSQL save operations
  - Use `config.py` for all paths
- [ ] **1.5**: Create train/test/holdout splits in Parquet
- [ ] **1.6**: Test on small sample, then full dataset
- [ ] **1.7**: Archive `notebooks/1_ETL_Spark.ipynb` (superseded by script)
- [ ] **1.8**: Update documentation with new ETL flow

**Outputs**:
- `data/processed/01_etl_output/train.parquet`
- `data/processed/01_etl_output/test.parquet`
- `data/processed/01_etl_output/holdout.parquet`
- `src/utils/time_discount.py`

---

### Stage 2.1: NLP - Basic Text Features

**Current State**: `2.1_NLP_Basic_Text_Processing_Spark.py` (204 lines) reads/writes PostgreSQL

**Goal**: Refactor to read/write Parquet, extract reusable utilities

**Tasks**:
- [ ] **2.1.1**: Read current script and understand text processing logic
- [ ] **2.1.2**: Refactor script:
  - Read from `01_etl_output/train.parquet`
  - Process: token counts, readability, sentiment (TextBlob)
  - Write to `data/processed/02_nlp_basic/train.parquet`
  - Use config.py for paths
  - Remove PostgreSQL dependencies
- [ ] **2.1.3**: Create `src/utils/spark_utils.py` for Spark session setup
- [ ] **2.1.4**: Test on sample then full dataset
- [ ] **2.1.5**: Archive `notebooks/2.1_NLP_Spark_Text_Basic.ipynb`
- [ ] **2.1.6**: Update pipeline documentation

**Outputs**:
- `data/processed/02_nlp_basic/train.parquet`
- `data/processed/02_nlp_basic/test.parquet`
- `src/utils/spark_utils.py`

---

### Stage 2.2: NLP - Linguistic Features (spaCy)

**Current State**: `2.2_NLP_Spacy_POS_ENT_DEP.py` (236 lines) reads/writes PostgreSQL

**Goal**: Refactor for Parquet, optimize chunked processing

**Tasks**:
- [ ] **2.2.1**: Understand spaCy processing logic (POS, NER, dependencies)
- [ ] **2.2.2**: Refactor script:
  - Read from `01_etl_output/train.parquet`
  - Process in chunks (100K records)
  - Write to `data/processed/02_nlp_spacy/train.parquet`
  - Use config.py
- [ ] **2.2.3**: Test and validate POS/NER/DEP features
- [ ] **2.2.4**: Archive `notebooks/2.2_NLP_Spacy_POS_ENT_DEP.ipynb`
- [ ] **2.2.5**: Update documentation

**Outputs**:
- `data/processed/02_nlp_spacy/train.parquet`
- `data/processed/02_nlp_spacy/test.parquet`

---

### Stage 2.3: NLP - TF-IDF + Classification Models

**Current State**: 
- `2.3.1_NLP_Spark_Tf-Idf_Models.py` (252 lines) - Spark version
- `notebooks/2.3_NLP_Tf-Idf_Models.ipynb` - Non-Spark version
- `notebooks/2.3.1_NLP_Spark_Tf-Idf_Models.ipynb` - Spark notebook

**Goal**: Consolidate to single production script, archive notebooks

**Tasks**:
- [ ] **2.3.1**: Review both script and notebooks for differences
- [ ] **2.3.2**: Refactor production script:
  - Read from `01_etl_output/train.parquet`
  - Train TF-IDF + Naive Bayes + SVM
  - Save models to `models/nlp/NB_TFIDF_all/`, `SVM_TFIDF_all/`
  - Write predictions to `data/processed/02_nlp_tfidf/train.parquet`
  - Use config.py
- [ ] **2.3.3**: Test model training and predictions
- [ ] **2.3.4**: Archive both notebooks to `notebooks/archive/development/`
- [ ] **2.3.5**: Document TF-IDF approach in README

**Outputs**:
- `data/processed/02_nlp_tfidf/train.parquet` (with NB_prob, svm_pred)
- `models/nlp/NB_TFIDF_all/` (retrained)
- `models/nlp/SVM_TFIDF_all/` (retrained)

---

### Stage 2.4: NLP - Word Embeddings

**Current State**:
- `2.4.1_NLP_Spark_Text_Embeddings.py` (356 lines) - Spark NLP
- `2.4.2_NLP_Fasttext.py` (109 lines) - FastText
- `notebooks/2.4_NLP_Embeddings.ipynb` - Non-Spark
- `notebooks/2.4.1_NLP_Spark_Embeddings.ipynb` - Spark

**Goal**: Consolidate embedding approaches, decide on primary method

**Tasks**:
- [ ] **2.4.1**: Compare Spark NLP vs FastText performance
- [ ] **2.4.2**: Decide: Keep both or consolidate to one?
- [ ] **2.4.3**: Refactor chosen approach(es):
  - Read from `01_etl_output/train.parquet`
  - Train embeddings
  - Write predictions to `data/processed/02_nlp_embeddings/train.parquet`
  - Save models to `models/nlp/`
- [ ] **2.4.4**: Archive non-Spark notebook
- [ ] **2.4.5**: Update embedding documentation

**Outputs**:
- `data/processed/02_nlp_embeddings/train.parquet` (with ft_prob)
- `models/nlp/fasttext_model_ALL/` (retrained)

---

### Stage 2.5: NLP - Topic Modeling (LDA)

**Current State**: 
- `2.5_NLP_Topic_Modeling_LDA.py` (174 lines)
- `notebooks/2.5_NLP_Topic_Modeling_LDA.ipynb` (1.5 MB with visualizations)

**Goal**: Refactor script, keep notebook for topic visualization

**Tasks**:
- [ ] **2.5.1**: Refactor script:
  - Read from `01_etl_output/train.parquet`
  - Preprocess with spaCy + NLTK
  - Train LDA (5 topics)
  - Write topic probabilities to `data/processed/02_nlp_lda/train.parquet`
  - Save model to `models/nlp/LDA_model_ALL/`
- [ ] **2.5.2**: Update notebook to read from Parquet, keep for visualization
- [ ] **2.5.3**: Test LDA training and topic assignment
- [ ] **2.5.4**: Document topic modeling approach

**Outputs**:
- `data/processed/02_nlp_lda/train.parquet` (with lda_t1-t5)
- `models/nlp/LDA_model_ALL/` (retrained)
- Keep: `notebooks/2.5_NLP_Topic_Modeling_LDA.ipynb` (visualization)

---

### Stage 3: Feature Combination

**Current State**:
- `3_ETL_Combine_Processed_Text_Data.sql` (401 lines) - SQL joins
- `notebooks/3.1_ETL_Combined_Data_to_CSV.ipynb` - Export utility
- `notebooks/3.2_EDA_Processed_Text_Data.ipynb` - EDA (2.9 MB)

**Goal**: Convert SQL to PySpark, combine all NLP features, export to CSV

**Tasks**:
- [ ] **3.1**: Convert SQL logic to PySpark script `3_Combine_Features.py`:
  - Read all processed Parquet files (01_etl, 02_nlp_*)
  - Join on review_id
  - Combine all features (80+ columns)
  - Write to `data/processed/03_combined/train.parquet`
  - Export to CSV: `data/model_ready/train.csv`
- [ ] **3.2**: Test feature combination logic
- [ ] **3.3**: Validate all features present (80+ columns)
- [ ] **3.4**: Archive `3_ETL_Combine_Processed_Text_Data.sql`
- [ ] **3.5**: Archive `notebooks/3.1_ETL_Combined_Data_to_CSV.ipynb`
- [ ] **3.6**: Keep `notebooks/3.2_EDA_Processed_Text_Data.ipynb` for analysis
- [ ] **3.7**: Update feature documentation

**Outputs**:
- `data/processed/03_combined/train.parquet` (all features)
- `data/model_ready/train.csv` (for ML compatibility)
- `data/model_ready/test.csv`
- `data/model_ready/feature_names.txt` (column list)

---

### Stage 4: Machine Learning - Base Models

**Current State**:
- `4.1_ML_Base_Models.py` (336 lines) - Trains 4 base models
- `notebooks/4.0_AutoML_PyCaret.ipynb` - AutoML experiments

**Goal**: Refactor base model training, keep AutoML notebook

**Tasks**:
- [ ] **4.1**: Create `src/utils/model_utils.py` with Base_Model_Process class
- [ ] **4.2**: Refactor `4.1_ML_Base_Models.py`:
  - Read from `data/model_ready/train.csv`
  - Use config.py for paths
  - Import Base_Model_Process from utils
  - Train: LogReg, DecisionTree, RandomForest, XGBoost
  - Save to `models/base_models/`
  - Use MLflow tracking
- [ ] **4.3**: Test on sample (100K rows) then full dataset
- [ ] **4.4**: Keep `notebooks/4.0_AutoML_PyCaret.ipynb` for experiments
- [ ] **4.5**: Document base model results

**Outputs**:
- `models/base_models/*.joblib` (4 models retrained)
- `src/utils/model_utils.py`

---

### Stage 5: Machine Learning - Dimensionality Reduction

**Current State**:
- `5_PCA_Dimensionality_Reduction.py` (290 lines)
- `notebooks/5.0_PCA_Dimensionality_Reduction.ipynb` (141 KB)
- `notebooks/5.1_Feature_Selection.ipynb` (447 KB)

**Goal**: Refactor PCA script, keep selection notebooks for analysis

**Tasks**:
- [ ] **5.1**: Refactor PCA script:
  - Read from `data/model_ready/train.csv`
  - Use config.py
  - Run PCA analysis
  - Save PCA-transformed datasets (optional)
- [ ] **5.2**: Keep both notebooks for feature analysis/visualization
- [ ] **5.3**: Document PCA approach and results

**Outputs**:
- Optional: `data/model_ready/train_pca.csv`
- Keep: Both notebooks for analysis

---

### Stage 6: Machine Learning - Final Models

**Current State**:
- `6.1_ML_Logistic_Regression.py` (637 lines) - Classification
- `10.1_ML_Linear_Regression.py` (626 lines) - Regression
- `notebooks/7.0_REG_Target_Adjustments.ipynb` - Target engineering
- `notebooks/8.0_AutoML_PyCaret_Regression.ipynb` - Regression AutoML
- `notebooks/9.1_Feature_Selection_Regression.ipynb` - Regression features

**Goal**: Refactor final model training scripts

**Tasks**:
- [ ] **6.1**: Refactor `6.1_ML_Logistic_Regression.py`:
  - Read from `data/model_ready/train.csv`
  - Use config.py
  - Cross-validation tuning
  - Save to `models/final_models/`
  - MLflow tracking
- [ ] **6.2**: Refactor `10.1_ML_Linear_Regression.py`:
  - Same refactoring pattern
  - Regression target handling
- [ ] **6.3**: Keep all 3 notebooks for experimentation/analysis
- [ ] **6.4**: Document final model performance

**Outputs**:
- `models/final_models/log_reg_*.joblib` (retrained)
- `models/final_models/lin_reg_*.joblib` (retrained)
- Keep: Target engineering and AutoML notebooks

---

### Stage 7: Predictions & Evaluation

**Current State**:
- `11.1_ETL_Add_Predictions.py` (600 lines) - Add predictions to dataset
- `notebooks/12.1_Review_Ranking.ipynb` - Ranking analysis

**Goal**: Refactor prediction script, keep ranking notebook

**Tasks**:
- [ ] **7.1**: Refactor `11.1_ETL_Add_Predictions.py`:
  - Read from `data/model_ready/train.csv`
  - Load final models from `models/final_models/`
  - Generate predictions
  - Save to `data/final_predict/`
  - Use config.py
- [ ] **7.2**: Keep `notebooks/12.1_Review_Ranking.ipynb` for analysis
- [ ] **7.3**: Update to read from new Parquet-based outputs
- [ ] **7.4**: Document final ranking methodology

**Outputs**:
- `data/final_predict/train_predictions.csv`
- `data/final_predict/test_predictions.csv`
- Keep: Ranking notebook

---

### Stage 8: Final Cleanup & Documentation

**Goal**: Polish codebase and update all documentation

**Tasks**:
- [ ] **8.1**: Run `black` formatter on all .py files
- [ ] **8.2**: Add docstrings to all refactored functions
- [ ] **8.3**: Update README.md with new pipeline flow
- [ ] **8.4**: Update all docs/ files affected by refactoring
- [ ] **8.5**: Create `notebooks/archive/development/` and move archived notebooks
- [ ] **8.6**: Remove all commented-out code
- [ ] **8.7**: Final git commit with summary

**Outputs**:
- Clean, documented codebase
- Updated README and documentation
- Organized notebook structure
  - Feature engineering functions
- [ ] Create test data fixtures (small samples)
- [ ] Add integration test for basic pipeline

#### Phase 8: Documentation Updates
- [ ] Update README.md with current state
  - Remove outdated information
  - Add "Getting Started" section
  - Document refactored structure
- [ ] Create CONTRIBUTING.md with development guidelines
- [ ] Add inline code comments for complex logic
- [ ] Document time-discounting methodology clearly
- [ ] Create data schema documentation
- [ ] Add architecture diagram to docs/

---

## 📋 BACKLOG - PHASE 1 (2020-Era Improvements)

### Performance Optimization for 2020-Era Code

**Goal**: Improve performance of existing pipeline without changing methodology  
**Why**: Python UDFs in Spark are 10-100x slower than native operations. Processing 8M reviews with current UDFs could take 4-6 hours.

**High-Level Approach**:
- Replace Python UDFs with Pandas UDFs (vectorized operations)
- Optimize Spark configurations for large dataset processing
- Add caching/persistence at key pipeline stages
- Profile code to identify bottlenecks
- Benchmark improvements (target: <1 hour for full pipeline)

**Priority**: Medium (do after Stage 1-3 refactoring complete)

---

### Improve Time Discounting Accuracy

**Goal**: Use actual business/user start dates instead of proxies  
**Why**: Current approach uses Yelp founding date (2004) for all businesses, creating inaccurate features. Real business start = first review date.

**High-Level Approach**:
- Calculate business start date from minimum review date per business
- Add to business Parquet file as `business_first_review_date`
- Update time discount formulas to use actual dates
- Document remaining assumptions (user average stars)
- Compare accuracy of improved vs original approach

**Priority**: Low (methodology improvement, not critical for refactoring)

---

### Create Comprehensive Testing Suite

**Goal**: Add unit tests, integration tests, and data validation  
**Why**: Large dataset and complex pipeline need automated testing to catch issues early.

**High-Level Approach**:
- Unit tests for time discount calculations
- Integration tests for each pipeline stage
- Data validation tests (schema, row counts, null checks)
- Regression tests for model predictions
- pytest fixtures with small sample datasets

**Priority**: Medium (after Stage 1-3 complete)

---

## 📋 BACKLOG - PHASE 2 (SOTA Rebuild)

### Modern Transformer-Based Approach (2025 SOTA)

**Goal**: Rebuild entire pipeline using modern NLP/ML techniques and compare with 2020-era approach  
**Why**: Demonstrate understanding of both classical and modern ML. Show evolution of NLP methods. Quantify improvements from SOTA techniques.

**High-Level Approach**:

**1. Data Layer** (Keep Parquet, simplify features):
- Use only review text + metadata (no complex feature engineering)
- Filter to recent reviews (2018-2020) to avoid time discounting complexity
- Simple train/test split, minimal preprocessing

**2. Text Embeddings** (Replace TF-IDF/FastText/LDA):
- Sentence transformers: `all-MiniLM-L6-v2` (384-dim embeddings)
- OR fine-tuned BERT: `bert-base-uncased` for classification
- Compare embedding quality vs. traditional approaches

**3. Model Architecture** (Replace sklearn models):
- Simple neural network on embeddings (2-3 layers)
- OR logistic regression on embeddings (simpler baseline)
- OR fine-tune BERT end-to-end (most modern)

**4. Comparison Framework**:
- Same evaluation metrics as 2020 approach (AUC, F1, accuracy)
- Compare training time (expect 10-100x faster)
- Compare model complexity (384 dims vs 80+ features)
- Compare performance (expect +5-15% AUC improvement)
- Document resource requirements (GPU vs CPU)

**5. Documentation**:
- Side-by-side comparison table
- Evolution of NLP methods: TF-IDF → Word2Vec → FastText → BERT
- Lessons learned from both approaches
- When to use which approach

**Deliverables**:
- `src/modern/` directory with new pipeline
- Comparison notebook: `notebooks/Phase2_Classical_vs_Modern.ipynb`
- Updated README with both approaches documented
- Blog post or presentation on classical vs modern NLP

**Timeline**: After Phase 1 complete (2-3 weeks of work)

**Tech Stack**:
- `sentence-transformers` or `transformers` (HuggingFace)
- `torch` or `tensorflow`
- `wandb` for experiment tracking
- `optuna` for hyperparameter tuning

---

### Few-Shot Learning with LLMs (Experimental)

**Goal**: Explore using GPT-4/Claude for zero-shot or few-shot review quality prediction  
**Why**: Cutting-edge approach, minimal training data needed, could work for new domains.

**High-Level Approach**:
- Design prompt for review quality classification
- Test with 0-shot, 5-shot, 10-shot examples
- Compare cost vs accuracy with fine-tuned BERT
- Evaluate for production feasibility
- Document findings

**Priority**: Low (experimental, after Phase 2 complete)

---

### Original Backlog Items (Updated Context)

### Create Configuration-Driven Pipeline

**Goal**: Make entire data pipeline reproducible and configurable without code changes  
**Why**: Currently requires manual code edits to change parameters, paths, or model configs. Need YAML/JSON config files for experimentation.

**High-Level Approach**:
- Create YAML config files for each pipeline stage
- Implement config validation
- Add CLI interface for running pipeline stages
- Enable experiment tracking via config versioning
- Document all configurable parameters

---

### Containerize Application

**Goal**: Create Docker containers for development and deployment  
**Why**: Eliminate "works on my machine" issues, simplify deployment, enable cloud-native scaling.

**High-Level Approach**:
- Create Dockerfile for training environment
- Create separate container for inference API
- Use docker-compose for local development
- Document container usage
- Optimize image size

---

### Build Prediction API

**Goal**: Create REST API for real-time review quality prediction  
**Why**: Enable integration with applications, demonstrate production-ready ML system.

**High-Level Approach**:
- Choose framework (FastAPI or Flask)
- Design API endpoints (predict, batch predict, health check)
- Implement model serving with caching
- Add API documentation (Swagger/OpenAPI)
- Containerize API
- Add authentication and rate limiting

---

### Implement MLOps Pipeline

**Goal**: Automate model training, evaluation, and deployment with CI/CD  
**Why**: Manual model training and deployment is error-prone and time-consuming. Need automated workflow for model lifecycle.

**High-Level Approach**:
- Set up MLflow for experiment tracking and model registry
- Create automated training pipeline
- Implement model validation and A/B testing
- Add monitoring and alerting
- Create automated retraining triggers
- Document MLOps processes

---

### Optimize for Large-Scale Processing

**Goal**: Improve performance and scalability for processing full 8M+ review dataset  
**Why**: Current pipeline uses samples (100K-1M). Need to efficiently process full dataset and scale to future growth.

**High-Level Approach**:
- Profile code to identify bottlenecks
- Optimize Spark configurations
- Implement data partitioning strategies
- Use incremental learning for models
- Explore Dask as Spark alternative
- Benchmark processing times

---

### Add Advanced Features

**Goal**: Enhance model performance with additional engineered features  
**Why**: Current features are comprehensive but could be enriched with user/business interaction patterns and temporal trends.

**High-Level Approach**:
- User-business interaction features (repeat reviewers)
- Temporal patterns (review velocity, seasonal trends)
- Geographic features (neighborhood clustering)
- Review similarity metrics
- Business category embeddings
- User reviewer tier/reputation scores

---

### Create Interactive Dashboard

**Goal**: Build web dashboard for exploring predictions and model performance  
**Why**: Enable stakeholders to understand model behavior, explore predictions, and provide feedback without code.

**High-Level Approach**:
- Choose framework (Streamlit, Dash, or Gradio)
- Create model performance visualizations
- Add review exploration interface
- Implement filtering and search
- Show feature importance and explanations
- Deploy dashboard to cloud

---

### Deep Learning Exploration

**Goal**: Experiment with deep learning architectures for review quality prediction  
**Why**: Traditional ML models work well, but deep learning (transformers, LSTMs) might capture more nuanced patterns in text.

**High-Level Approach**:
- Fine-tune BERT/RoBERTa for classification
- Experiment with multi-task learning (useful/funny/cool separately)
- Try ensemble of traditional + deep learning
- Compare performance vs. computational cost
- Document findings and recommendations

---

## ✅ COMPLETED WORK

### Project Analysis & Documentation Setup (December 19, 2025)

**Achievements**:
- Conducted comprehensive project exploration
  - Reviewed all documentation files (README, Action Plan, proposals)
  - Analyzed data pipeline architecture and directory structure
  - Examined 20+ notebooks across ETL, NLP, and ML stages
  - Reviewed 15+ source code scripts
  - Cataloged models directory (base models, final models, NLP models)
  - Identified 8533 model experiments in version_1/model_info.csv
- Created `.github/copilot-instructions.md` - comprehensive agent instructions file for GitHub Copilot
  - Documented project overview, goals, and business value
  - Cataloged technology stack (15+ Python libraries, Spark, PostgreSQL, AWS)
  - Mapped complete data pipeline architecture
  - Defined project structure and file naming conventions
  - Documented key concepts (time discounting, target creation, feature engineering)
  - Identified code quality issues and refactoring priorities
  - Established coding standards and development workflow
- Created `TODO.md` - structured project management file
  - Defined active sprint: Codebase Cleanup & Foundation Refactor
  - Broke down sprint into 8 detailed phases with specific subtasks
  - Created backlog with 8 high-level improvement initiatives
  - Established project management approach
- Identified critical technical debt
  - Hardcoded file paths (EC2 vs. local) throughout codebase
  - Duplicate code between notebooks and .py scripts
  - No centralized configuration management
  - Limited error handling and logging
  - Inconsistent naming conventions
  - Version 0 and version 1 code coexisting
- Documented data flow
  - JSON (10GB) → Spark ETL → PostgreSQL → NLP Processing → Feature Engineering → ML Models → Predictions
  - Identified 5 original Yelp dataset files
  - Mapped processed data locations (model_ready, final_predict)
  - Cataloged 4 base models, multiple final models, and NLP models

**Key Findings**:
- Project successfully predicts review quality with ~90% accuracy and 0.96+ AUC
- Complex time-discounting methodology to handle temporal data issues
- Extensive feature engineering: 80+ features from text analysis (NLP), user metadata, and business data
- Well-documented academic project but needs refactoring for production use
- Strong foundation with Spark, PostgreSQL, MLflow already in place

---
