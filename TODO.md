# Yelp Review Quality Prediction - TODO

> **Last Updated**: December 19, 2025  
> **Current Sprint**: Codebase Cleanup & Foundation Refactor
>
> **Project Management Approach**: Active sprint contains expansive planning with phases and subtasks. Backlog items remain high-level (goal + why + approach). Completed sprints condensed to title, date, and key achievements only.

---

## 🎯 ACTIVE SPRINT

### Codebase Cleanup & Foundation Refactor

**Goal**: Clean up legacy codebase, remove dead code, establish consistent patterns, and create foundation for future development  
**Why**: Project is an old academic capstone (2020-2021) with hardcoded paths, duplicated code, inconsistent naming, and no centralized configuration. Must establish clean foundation before adding features or modernizing.

**High-Level Approach**:
- Audit entire codebase to identify dead code, duplicates, and inconsistencies
- Create centralized configuration system for paths and parameters
- Establish consistent coding standards and documentation patterns
- Separate exploration code (notebooks) from production code (scripts)
- Remove or archive deprecated version_0 and version_1 code
- Set up proper logging and error handling

**Phases**:

#### Phase 1: Discovery & Documentation
- [x] Create comprehensive .github/copilot-instructions.md file documenting project architecture
- [x] Create TODO.md for project management
- [ ] Map all file dependencies (which scripts depend on which data files)
- [ ] Inventory all hardcoded paths (EC2 vs local)
- [ ] Identify duplicate code between notebooks and .py files
- [ ] List deprecated or unused scripts
- [ ] Document current data pipeline flow with diagram
- [ ] Catalog all trained models and their purposes

#### Phase 2: Configuration Management
- [ ] Create `config.py` for centralized configuration
  - Environment detection (EC2 vs local vs Docker)
  - All file paths (data, models, logs)
  - Model hyperparameters
  - Spark configurations
  - Database connection settings (reference confidential.py)
- [ ] Create `requirements.txt` with pinned versions
- [ ] Create `environment.yml` for conda environments
- [ ] Update all scripts to use centralized config
- [ ] Add .env support for local development settings

#### Phase 3: Code Standards & Cleanup
- [ ] Run black formatter on all .py files
- [ ] Run flake8 and fix critical issues
- [ ] Add comprehensive docstrings to all functions and classes
  - Follow Google/NumPy docstring format
  - Include parameter types and return types
- [ ] Remove all commented-out code blocks
- [ ] Replace magic numbers with named constants
- [ ] Standardize variable naming conventions
  - Snake_case for functions/variables
  - PascalCase for classes
- [ ] Add type hints to function signatures

#### Phase 4: Dead Code Removal
- [ ] Archive or delete `src/version_0/` directory
- [ ] Review and consolidate `src/version_1/` code
- [ ] Remove duplicate functionality between notebooks and scripts
- [ ] Delete unused model artifacts from `models/version_1/`
- [ ] Clean up commented-out data export/save statements
- [ ] Remove obsolete imports

#### Phase 5: Logging & Error Handling
- [ ] Implement Python logging framework
  - Replace print statements with logging
  - Add log levels (DEBUG, INFO, WARNING, ERROR)
  - Configure log files by module
- [ ] Add try/except blocks for:
  - File I/O operations
  - Database connections
  - Model loading/saving
  - Spark operations
- [ ] Create utility functions for common error scenarios
- [ ] Add validation for data loading (check dtypes, required columns)

#### Phase 6: Code Organization
- [ ] Create `src/utils/` directory for shared utilities
  - `data_loader.py` - Centralized data loading with path management
  - `model_utils.py` - Model save/load/evaluation utilities
  - `feature_engineering.py` - Shared feature engineering functions
  - `spark_utils.py` - Spark session creation and helpers
  - `time_discount.py` - Time discounting calculations
- [ ] Refactor Base_Model_Process class into `src/utils/model_utils.py`
- [ ] Create clear separation:
  - `src/etl/` - ETL scripts (1.x)
  - `src/nlp/` - NLP processing (2.x)
  - `src/modeling/` - ML training (4.x-11.x)
  - `src/evaluation/` - Model evaluation (12.x)
- [ ] Update imports across all files

#### Phase 7: Testing Foundation
- [ ] Create `tests/` directory structure
- [ ] Add pytest configuration
- [ ] Write unit tests for utility functions
  - Time discounting calculations
  - Data validation
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

## 📋 BACKLOG

### Modernize NLP Pipeline

**Goal**: Replace older NLP libraries and approaches with modern transformers and sentence embeddings  
**Why**: Project uses 2020-era NLP (spaCy, Gensim LDA, fastText). Modern transformers (BERT, RoBERTa, SBERT) offer better performance and are industry standard.

**High-Level Approach**:
- Evaluate Hugging Face transformers for text classification
- Replace fastText with sentence-transformers for embeddings
- Compare performance vs. existing models
- Update feature engineering pipeline
- Benchmark memory/compute requirements

---

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
