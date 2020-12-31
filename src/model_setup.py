
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import (ElasticNet,
                                  LogisticRegression,
                                  SGDClassifier,
                                  SGDRegressor)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (RandomForestClassifier,
                              RandomForestRegressor,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from xgboost.sklearn import XGBRegressor, XGBClassifier


class ModelSetupInfo():
    """
    Storage class for model hyperparameters and
    other setup info.
    For param dicts: All values must be lists for CV.
    Will work fine with a list with a single value.
    """
    def __init__(self):

        # PREPROCESSING
        self.scalars = {'Standard': StandardScaler,
                        'Power': PowerTransformer}

        # REGRESSION
        self.reg_models = \
            {'Elastic Net': ElasticNet,
             'Forest Reg': RandomForestRegressor,
             'HGB Reg': HistGradientBoostingRegressor,
             'XGB Reg': XGBRegressor}
        self.reg_params_cv = \
            {'Elastic Net': {'alpha': [0.5, 1, 1.5, 3],
                             'l1_ratio': [0, 0.25, 0.5, 0.75, 1],
                             'tol': [0.0001],
                             'max_iter': [100, 1000, 10000],
                             'random_state': [7]},
             'Forest Reg': {'n_estimators': [5, 10],
                            'criterion': ['mse', 'mae'],
                            'max_depth': [None, 10, 100],
                            'min_samples_split': [2, 10],
                            'min_samples_leaf': [1, 5],
                            'max_features': [None, 'sqrt', 10],
                            'max_leaf_nodes': [None, 10, 100],
                            'random_state': [7],
                            'max_samples': [None, 0.1, 0.5]},
             'HGB Reg': {'loss': ['least_squares', 'least_absolute_deviation',
                                  'poisson'],
                         'learning_rate': [0.1, 0.4, 0.7],
                         'max_iter': [10, 100, 1000],
                         'l2_regularization': [0, 0.25, 0.5, 0.75],
                         'max_depth': [10, 50, 100],
                         'max_leaf_nodes': [31],
                         'min_samples_leaf': [20],
                         },
             'XGB Reg': {'learning_rate': [0.05, 0.1, 0.25],
                         'n_estimators': [10, 100, 1000, 10000],
                         'max_depth': [3, 5, 8, 10],
                         'min_child_weight': [1, 3, 5],
                         'gamma': [0, 0.2, 0.5],
                         'subsample': [0.5, 0.8],
                         'colsample_bytree': [0.5, 0.8],
                         'reg_lambda': [0, 0.25, 0.5, 0.75],
                         'random_state': [7]}
             }
        self.reg_params = \
            {'Elastic Net': {'alpha': [1],
                             'l1_ratio': [0.5],
                             'tol': [0.0001],
                             'max_iter': [1000],
                             'random_state': [7]},
             'Forest Reg': {'n_estimators': [10],
                            'criterion': ['mse'],
                            'max_depth': [100],
                            'min_samples_split': [2],
                            'min_samples_leaf': [1],
                            'max_features': ['sqrt'],
                            'max_leaf_nodes': [100],
                            'random_state': [7],
                            'max_samples': [0.5]},
             'HGB Reg': {'loss': ['least_squares'],
                         'learning_rate': [0.4],
                         'max_iter': [100],
                         'l2_regularization': [0.25],
                         'max_depth': [50],
                         'max_leaf_nodes': [31],
                         'min_samples_leaf': [20],
                         },
             'XGB Reg': {'learning_rate': [0.1],
                         'n_estimators': [1000],
                         'max_depth': [10],
                         'min_child_weight': [3],
                         'gamma': [0.2],
                         'subsample': [0.8],
                         'colsample_bytree': [0.8],
                         'reg_lambda': [0.5],
                         'random_state': [7]}
             }
        self.reg_scoring = \
            {'R2 Score': 'r2',
             'MSE': 'neg_mean_squared_error',
             'RMSE': 'neg_root_mean_squared_error',
             'MAE': 'neg_mean_absolute_error'}

        # CLASSIFICATION
        self.cls_models = \
            {'Log Reg': LogisticRegression,
             'Forest Cls': RandomForestClassifier,
             'HGB Cls': HistGradientBoostingClassifier,
             'XGB Cls': XGBClassifier}
        self.cls_params_cv = \
            {'Log Reg': {'penalty': ['elasticnet'],
                         'tol': [0.0001],
                         'C': [0.5, 1.0, 2],
                         'class_weight': [None, 'balanced'],
                         'l1_ratio': [0, 0.25, 0.5, 0.75, 1],
                         'random_state': [7],
                         'solver': ['saga'],
                         'max_iter': [100, 1000, 10000]},
             'Forest Cls': {'n_estimators': [5, 10],
                            'criterion': ['gini', 'entropy'],
                            'max_depth': [None, 10, 100],
                            'min_samples_split': [2, 10],
                            'min_samples_leaf': [1, 5],
                            'max_features': [None, 'sqrt', 10],
                            'max_leaf_nodes': [None, 10, 100],
                            'class_weight': [None, 'balanced'],
                            'max_samples': [None, 0.1, 0.5],
                            'random_state': [7]},
             'HGB Cls': {'loss': ['auto'],
                         'learning_rate': [0.1, 0.4, 0.7],
                         'max_iter': [10, 100, 1000],
                         'max_leaf_nodes': [31],
                         'min_samples_leaf': [20],
                         'l2_regularization': [0, 0.25, 0.5, 0.75],
                         'random_state': [7]},
             'XGB Cls': {'learning_rate': [0.05, 0.1, 0.25],
                         'n_estimators': [10, 100, 1000, 10000],
                         'max_depth': [3, 5, 8, 10],
                         'min_child_weight': [1, 3, 5],
                         'gamma': [0, 0.2, 0.5],
                         'subsample': [0.5, 0.8],
                         'colsample_bytree': [0.5, 0.8],
                         'reg_lambda': [0, 0.25, 0.5, 0.75],
                         'random_state': [7]}
             }
        self.cls_params = \
            {'Log Reg': {'penalty': ['elasticnet'],
                         'tol': [0.0001],
                         'C': [0.5, 1.0, 2],
                         'class_weight': ['balanced'],
                         'l1_ratio': [0.25],
                         'random_state': [7],
                         'solver': ['saga'],
                         'max_iter': [1000]},
             'Forest Cls': {'n_estimators': [10],
                            'criterion': ['gini'],
                            'max_depth': [100],
                            'min_samples_split': [2],
                            'min_samples_leaf': [1],
                            'max_features': ['sqrt'],
                            'max_leaf_nodes': [100],
                            'class_weight': ['balanced'],
                            'max_samples': [None],
                            'random_state': [7]},
             'HGB Cls': {'loss': ['auto'],
                         'learning_rate': [0.4],
                         'max_iter': [100],
                         'max_leaf_nodes': [31],
                         'min_samples_leaf': [20],
                         'l2_regularization': [0.25],
                         'random_state': [7]},
             'XGB Cls': {'learning_rate': [0.1],
                         'n_estimators': [100],
                         'max_depth': [10],
                         'min_child_weight': [3],
                         'gamma': [0.2],
                         'subsample': [0.8],
                         'colsample_bytree': [0.8],
                         'reg_lambda': [0.5],
                         'random_state': [7]}
             }
        self.cls_scoring = \
            {'Accuracy': 'accuracy',
             'Balanced Accuracy': 'balanced_accuracy',
             'Precision': 'precision_weighted',
             'Recall': 'recall_weighted',
             'F1_Score': 'f1_weighted',
             'ROC_AUC': 'roc_auc_ovr_weighted'}

        # NLP
        # Naive Bayes and Support Vector Machines are common in NLP.
        # Can also use forests, boosting, and neural nets.
        self.nlp_models = \
            {'sgd_cls': SGDClassifier,
             'sgd_reg': SGDRegressor,
             'naive_bayes': MultinomialNB}
        self.nlp_params_cv = \
            {'sgd_cls': {'loss': ['hinge', 'log', 'modified_huber',
                                  'squared_hinge', 'perceptron'],
                         'penalty': ['elasticnet'],
                         'l1_ratio': [0, 0.15, 0.35, 0.5, 0.75],
                         'alpha': [0.00001, 0.0001, 0.001, 0.1],
                         'random_state': [7],
                         'max_iter': [10, 100, 1000],
                         'class_weight': [None, 'balanced']},
             'sgd_reg': {'loss': ['squared_loss', 'huber',
                                  'epsilon_insensitive',
                                  'squared_epsilon_insensitive'],
                         'penalty': ['elasticnet'],
                         'l1_ratio': [0, 0.15, 0.35, 0.5, 0.75],
                         'alpha': [0.00001, 0.0001, 0.001, 0.1],
                         'random_state': [7],
                         'max_iter': [10, 100, 1000]},
             'naive_bayes': {'alpha': [0.1, 0.5, 1, 2]}}
        self.nlp_params = \
            {'sgd_cls': {'loss': ['hinge'],
                         'penalty': ['elasticnet'],
                         'l1_ratio': [0.15],
                         'alpha': [0.0001],
                         'random_state': [7],
                         'max_iter': [100],
                         'class_weight': ['balanced']},
             'sgd_reg': {'loss': ['squared_loss'],
                         'penalty': ['elasticnet'],
                         'l1_ratio': [0.15],
                         'alpha': [0.0001],
                         'random_state': [7],
                         'max_iter': [100]},
             'naive_bayes': {'alpha': [1]}}


if __name__ == "__main__":
    pass
