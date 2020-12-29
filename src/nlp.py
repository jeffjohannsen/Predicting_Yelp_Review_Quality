
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (RandomizedSearchCV,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import (ElasticNet,
                                  LogisticRegression,
                                  SGDClassifier,
                                  SGDRegressor)
from sklearn.ensemble import (RandomForestClassifier,
                              RandomForestRegressor,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.feature_extraction.text import (CountVectorizer,
                                             HashingVectorizer,
                                             TfidfTransformer,
                                             TfidfVectorizer)
from xgboost import XGBRegressor, XGBClassifier

from business_checkins import load_dataframe_from_yelp_2

if __name__ == "__main__":
    # load model info from csv
    model_info = pd.read_csv('../model_info.csv')
    # Keeping track of everything in csv file.
    model_info_record = {'question': None,
                         'data': None,
                         'record_count': None,
                         'goal': None,
                         'target': None,
                         'features': None,
                         'nlp_models': None,
                         'nlp_hyperparameters': None,
                         'model_type': None,
                         'hyperparameters': None,
                         'threshold': None,
                         'cost_function': None,
                         'results_metrics': None
                         }

    # load data
    def load_data(question, records):
        table = None
        if question == 'td':
            table = 'working_td_data'
        elif question == 'non_td':
            table = 'working_non_td_data'
        else:
            print('Invalid question argument')
            return None
        record_count = records
        query = f'''
                SELECT *
                FROM {table}
                LIMIT {record_count}
                ;
                '''
        model_info_record['question'] = question
        model_info_record['record_count'] = record_count
        return load_dataframe_from_yelp_2(query)

    def prep_data(df, datatype, target):
        data = None
        goal = None
        if datatype == 'text':
            data = df.loc[:, ['review_id', 'review_text',
                              'T1_REG_review_total_ufc',
                              'T2_CLS_ufc_>0', 'T3_CLS_ufc_level',
                              'T4_REG_ufc_TD', 'T5_CLS_ufc_level_TD',
                              'T6_REG_ufc_TDBD']]
        elif datatype == 'non_text':
            data = df.drop('review_text', axis=1)
        elif datatype == 'both':
            data = df
        else:
            print('Invalid datatype argument')
            return None
        
        reg_targets = ['T1_REG_review_total_ufc', 'T4_REG_ufc_TD',
                       'T6_REG_ufc_TDBD']
        cls_targets = ['T2_CLS_ufc_>0', 'T3_CLS_ufc_level',
                       'T5_CLS_ufc_level_TD']
        if target in cls_targets:
            goal == 'cls'
        elif target in reg_targets:
            goal == 'reg'
        else:
            print('Invalid target argument')
            return None
        
        target_data = data[target]
        features_data = data.drop(labels=(reg_targets + cls_targets), axis=1)
        model_info_record['data'] = datatype
        model_info_record['target'] = target
        model_info_record['features'] = features_data.columns
        model_info_record['goal'] = goal
        if goal == 'cls':
            return train_test_split(features_data, target_data, test_size=0.20,
                                    random_state=7, shuffle=True,
                                    stratify=target_data)
        elif goal == 'reg':
            return train_test_split(features_data, target_data, test_size=0.20,
                                    random_state=7, shuffle=True)

    # Train Test Split
    X_train, X_test, y_train, y_test = prep_data()

    # NLP
    # Lots of options for vectorizing. Spacy and NLTK expand the options here.
    # Default preprocessor, tokenizer, n-grams
    # CountVectorizer - https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    # HashingVectorizer - https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
    count_vectorizer_params = {'strip_accents': None,  # 'unicode'
                               'lowercase': True,  # False
                               'stop_words': 'english',  # others found online or made myself
                               'min_df': 1,  # 0 to 1
                               'max_df': 1,  # 0 to 1
                               'max_features': None,  # 1+
                               'ngram_range': (1, 1)}
    hashing_vectorizer_params = {'strip_accents': None,  # 'unicode'
                                 'lowercase': True,  # False
                                 'stop_words': 'english',  # others found online or made myself
                                 'min_df': 1,  # 0 to 1
                                 'max_df': 1,  # 0 to 1
                                 'n_features': (2 ** 20),
                                 'ngram_range': (1, 1),
                                 'norm': 'l2'}  # 'l1'
    vectorizer = [CountVectorizer(), HashingVectorizer()]
    
    # Tfidf
    normalizer_params = {'norm': 'l2'}  # 'l1'
    normalizer = TfidfTransformer()

    # Pipeline bringing all NLP together.
    text_pipeline = Pipeline([('vectorizer', vectorizer),
                              ('normalizer', normalizer),
                              ('model', NaiveBayesCLS)])

    # MAIN MODEL ------------------------------------------------------

    regression_models = {'Elastic Net': ElasticNet,
                         'Forest Reg': RandomForestRegressor,
                         'HGB Reg': HistGradientBoostingRegressor,
                         'XGB Reg': XGBRegressor}
    regression_model_params = {'Elastic Net': {'alpha': 1,
                                               'l1_ratio': 0.5,
                                               'tol': 0.0001,
                                               'max_iter': 1000,
                                               'random_state': 7},
                               'Forest Reg': {'n_estimators': 100,  # Number of trees
                                              'criterion': 'gini',  # 'entropy'
                                              'max_depth': None,
                                              'min_samples_split': 2,
                                              'min_samples_leaf': 1,
                                              'max_features': 'auto',
                                              'max_leaf_nodes': None,
                                              'random_state': 7,
                                              'class_weight': None,  # 'balanced'
                                              'max_samples': None},
                               'HGB Reg': {},
                               'XGB Reg': {}
                               }
    regression_scoring = {'R2 Score': 'r2',
                          'MSE': 'neg_mean_squared_error',
                          'RMSE': 'neg_root_mean_squared_error',
                          'MAE': 'neg_mean_absolute_error'}

    classification_models = {'Log Reg': LogisticRegression,
                             'Forest Cls': RandomForestClassifier,
                             'HGB Cls': HistGradientBoostingClassifier,
                             'XGB Cls': XGBClassifier}
    classification_model_params = {'Log Reg': {'penalty': 'elasticnet',
                                               'tol': 0.0001,
                                               'C': 1.0,
                                               'class_weight': None,  # 'balanced'
                                               'l1_ratio': None,  # Need to set this
                                               'random_state': 7,
                                               'solver': 'saga',  # Look into this
                                               'max_iter': 1000},
                                   'Forest Cls': {'n_estimators': 100,  # Number of trees
                                                  'criterion': 'mse',  # 'mae'
                                                  'max_depth': None,
                                                  'min_samples_split': 2,
                                                  'min_samples_leaf': 1,
                                                  'max_features': 'auto',
                                                  'max_leaf_nodes': None,
                                                  'random_state': 7,
                                                  'class_weight': None,  # 'balanced'
                                                  'max_samples': None},
                                   'HGB Cls': {'learning_rate': 0.1,
                                               'max_iter': 100,
                                               'max_leaf_nodes': 31,
                                               'min_samples_leaf': 20,
                                               'l2_regularization': 0,
                                               'random_state': 7},
                                   'XGB Cls': {}
                                   }
    classification_scoring = {'Accuracy': 'accuracy',
                              'Balanced Accuracy': 'balanced_accuracy',
                              'Precision': 'precision_weighted',
                              'Recall': 'recall_weighted',
                              'F1_Score': 'f1_weighted',
                              'ROC_AUC': 'roc_auc_ovr_weighted'}

    # Cross Validation
    # Example for classification.
    ModelCV = RandomizedSearchCV(classification_models['Forest Cls'](random_state=7),
                                 classification_model_params['Forest Cls'],
                                 n_iter=10,  # Number of hyperparameter combinations to try.
                                 scoring=classification_scoring,
                                 n_jobs=None,
                                 cv=None,  # Cross Validation technique. None defaults to 5-fold.
                                 refit='accuracy',  # Which of multiple scoring methods to use at the end for predictions, etc.
                                 random_state=7)  # For choosing hyperparameter combinations.
    ModelCV.fit(X_train, y_train)
    CV_results = pd.Dataframe(ModelCV.cv_results_)

    # Metrics



    # update model info

    # save model info to csv
    model_info.to_csv('../model_info.csv')