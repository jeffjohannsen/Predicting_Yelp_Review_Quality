# Yelp Review Quality - Project Outline and Action Plan

## Table of Contents
* [Project Goals](#Project-Goals)
    * [Central Questions](#Central-Questions)
    * [Methods and Concepts](#Methods-and-Concepts)
    * [Tech Used](#Tech-Used)
* [Current Focus](#Current-Focus)
* [Next Steps and Notes](#Next-Steps-and-Notes)
    * [Data Acquisition](#Data-Acquisition)
    * [Data Storage](#Data-Storage)
    * [Data Cleaning and Processing](#Data-Cleaning-and-Processing)
    * [Feature Engineering](#Feature-Engineering)
    * [Analysis and Visualization](#Analysis-and-Visualization)
    * [NLP](#NLP)
    * [Machine Learning](#Machine-Learning)
    * [Deployment and Production](#Deployment-and-Production)

# Project Goals

### Primary Goal 

* Create a prediction system that evaluates the quality (usefulness to user) of a review at the time the review is created.

### Secondary Goals

* Do a deeper dive into Yelp data analysis to locate more interesting questions and projects.
* Create a recommendation system to showcase the highest quality reviews to users.
* Deep dive into NLP to learn more advanced skills and tools.
* Deep dive into deep learning to add a level of knowledge in this area.
* Create a starting point and connection point for moving into Bizy project.

## Central Questions

1. Can the quality of a Yelp review be predicted at a level that is worthwhile to be implemented within the broader user interface and experience?
2. What factors provide the most information regarding the quality of a Yelp review?

## Methods and Concepts

ETL
* Distributed Computing
* MapReduce
EDA
NLP
* Text Readability, Sentiment, Polarity, and Subjectivity
* Topic Modeling
* Word and Sentence Embeddings
* Part of Speech, Named Entity, and Syntactic Dependency Analysis
ML
* AutoML
* Dimensionality Reduction
    * PCA
Recommendation Systems
Web App
Visualization and Presentation

## Tech Used

Code
* Python
    * Data Analysis - Pandas, Dataprep.eda, D-tale
    * Visualization - Matplotlib, Seaborn
    * Machine Learning - Sklearn, PyCaret, Mlflow
    * NLP - Spacy, Gensim, NLTK, WordCloud, TextBlob, Spark NLP
    * Other - PySpark, Numpy
* SQL 
    * SparkSQL
    * PostgreSQL
    * SQLAlchemy
    * psql
Tech
* AWS
    * S3
    * EC2
    * RDS
    * Aurora
    * EMR 
* Git/Github

# Current Focus

1. 
2.  
3. 

### Project Timeline Overview

~~NLP Feature Engineering~~ > Analysis and Visualization > ~~Baseline Modeling(LogReg)~~ > ~~Feature Selection and Extraction~~ > ~~Classification Modeling~~ > Target Time Discounting > Regression Modeling > Recommendation System and Custom Cost/Ranking Functions > Flask App and Plotly/Dash Dashboard > End Phase 2 > Non-Text Data Starting with improved feature time discounting

# Next Steps and Notes

## Data Acquisition

Complete

## Data Storage

Complete

## Data Cleaning and Processing

Time Discounting prior to Regression Problems

## Feature Engineering

Complete

## Analysis and Visualization

## NLP Feature Engineering

Complete

## Machine Learning


### Feature Selection and Extraction

Complete

### Classification Modeling:
Primary Focus
* Logistic Regression

### Regression Modeling:
Secondary Focus
* Linear Regression, Regularized Regression
* Decision Tree
* Random Forest
* XGBoost
* Neural Net

### Recommendation System and Custom Cost Functions

## Deployment and Production

* Flask App
    * Dashboard
    * User Interface for inputting reviews to be scored.
    * /score endpoint for showcasing the cls, reg, and rec-sys model pipelines  