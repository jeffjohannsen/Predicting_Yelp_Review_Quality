
# ! WARNING: Work in progress code that still needs docstrings and testing.
# TODO: Combine this pipeline with the other code for random forest.

import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import matplotlib.pyplot as plt


class ModelPipeline():

    def __init__(self):
        self.data = None
        self.target = None
        self.features = None
        self.scaled_features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, table_name, n_records=100000):
        connect = 'postgresql+psycopg2://postgres:password@localhost:5432/yelp'
        engine = create_engine(connect)
        query = f'''
                SELECT *
                FROM {table_name}
                LIMIT {n_records}
                ;
                '''
        df = pd.read_sql(query, con=engine)
        self.data = df.copy()

    def prepare_data(self, target_name, features_to_ignore):
        self.data = self.data.drop_duplicates(subset='review_id')
        self.target = self.data[target_name]
        self.features = self.data.drop(labels=features_to_ignore,
                                       axis=1,
                                       errors='ignore')

    def scale_features(self):
        scalar = StandardScaler()
        feature_names = self.features.columns
        self.features = pd.DataFrame(scalar.fit_transform(self.features),
                                     columns=feature_names)

    def split_data(self, test_size=0.20, random_state=5):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features,
                                                            self.target, test_size=test_size,
                                                            random_state=random_state)

    def run_logistic(self, random_state=5):
        log_reg = LogisticRegression(random_state=random_state)
        log_reg.fit(self.X_train, self.y_train)
        accuracy_score = log_reg.score(self.X_test, self.y_test)
        print(f'Test Accuracy Score: {accuracy_score:.3f}')
        coef = zip(list(self.features.columns), list(log_reg.coef_)[0])
        sorted_coef = sorted(coef, key=lambda x: x[1], reverse=True)
        print('Feature Coefficients:')
        for f, c in sorted_coef:
            print(f'{c:.3f} - {f}')

    def run_random_forest(self, params):
        forest = RandomForestClassifier(**params,
                                        random_state=5,
                                        oob_score=True,
                                        verbose=2
                                        )
        forest.fit(self.X_train, self.y_train)
        feature_importances = forest.feature_importances_
        oobscore = forest.oob_score_
        print(f'Out-of-Bag Score: {oobscore:.3f}')
        test_accuracy_score = forest.score(self.X_test, self.y_test)
        print(f'Test Accuracy: {test_accuracy_score:.3f}\n')
        ftrimpt = pd.Series(feature_importances,
                            index=self.features.columns).sort_values(ascending=False)
        print('Feature Importances:')
        print(ftrimpt.round(3))
        std = np.std([tree.feature_importances_
                     for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(feature_importances)
        # Plot the feature importances of the forest.
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Feature importances")
        ax.barh(range(self.features.shape[1]),
                feature_importances[indices],
                color="r",
                xerr=std[indices],
                align="center")
        feature_labels = self.features.columns[indices]
        ax.set_yticks(range(self.features.shape[1]))
        ax.set_yticklabels(feature_labels)
        ax.set_ylim([-1, self.features.shape[1]])
        plt.show()

    # TODO: Not fully tested yet.
    def plot_roc_curves(self):
        log_reg = LogisticRegression(**log_params, random_state=5)
        log_reg.fit(self.X_train, self.y_train)
        log_predict_proba = log_reg.predict_proba(self.X_test)
        log_reg_accuracy_score = log_reg.score(self.X_test, self.y_test)
        fpr_log, tpr_log, _ = roc_curve(self.y_test, log_predict_proba[:, [0]])
        log_auc = roc_auc_score(self.y_test, log_predict_proba[:, [0]])

        forest = RandomForestClassifier(**forest_params,
                                        random_state=5,
                                        oob_score=True,
                                        verbose=2
                                        )
        forest.fit(self.X_train, self.y_train)
        forest_predict_proba = forest.predict_proba(self.X_test)
        forest_accuracy_score = forest.score(self.X_test, self.y_test)
        fpr_forest, tpr_forest, _ = roc_curve(self.y_test,
                                              forest_predict_proba[:, [0]])
        forest_auc = roc_auc_score(self.y_test, forest_predict_proba[:, [0]])

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot([0, 1], [0, 1], 'k--')
        ax.plot(fpr_log, tpr_log, color='darkorange',
                label=f'Logistic Regression\n -Area under curve: {log_auc:.2f}\n -Accuracy: {log_reg_accuracy_score:.2f}')
        ax.plot(fpr_forest, tpr_forest, color='darkorange',
                label=f'Random Forest\n Area under curve = {forest_auc:.2f}\n -Accuracy: {forest_accuracy_score:.2f}')
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title('ROC curve')
        ax.legend(loc='best')
        plt.show()


def create_classification_data(df):
    data = df.copy()
    data['TARGET_review_has_upvotes'] = data['TARGET_review_upvotes_time_adjusted'].apply(lambda x: 1 if x > 0 else 0)
    data.drop('TARGET_review_upvotes_time_adjusted', axis=1, inplace=True)
    return data


def train_test_sql(df, sql_engine):
    data = df.copy()
    train_data, test_data = train_test_split(data, test_size=0.20,
                                             random_state=5)
    train_data.to_sql('model_data_cls_train', con=sql_engine, index=True,
                      if_exists='replace', chunksize=50000)
    test_data.to_sql('model_data_cls_test', con=sql_engine, index=True,
                     if_exists='replace', chunksize=50000)


if __name__ == "__main__":
    pd.set_option('display.max_columns', 100)
    pd.set_option("max_rows", 1000)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    unused_features = ['level_0', 'index', 'review_id',
                       'restaurant_latitude',
                       'restaurant_longitude',
                       'TARGET_review_has_upvotes']
    num_of_records = 100000

    # Always
    test = ModelPipeline()
    test.load_data('model_data_cls_train', num_of_records)
    test.prepare_data('TARGET_review_has_upvotes', unused_features)

    # Only scale for Logistic
    # test.scale_features()

    # Always
    test.split_data()

    # Logistic
    log_params = dict(penalty='l2',
                      C=1.0,
                      solver='lbfgs',
                      max_iter=100)

    # test.run_logistic()

    # Random Forest
    forest_params = dict(n_estimators=100,
                         criterion='entropy',
                         max_depth=None,
                         max_features='sqrt',
                         max_leaf_nodes=None,
                         min_samples_split=2,
                         min_samples_leaf=1,
                         min_impurity_decrease=0.0,
                         max_samples=None)

    # test.run_random_forest(params)

    # Both ROC Curves

    test.plot_roc_curves()
