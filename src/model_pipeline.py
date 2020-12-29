
import random
import string
import pprint
import pickle
import os.path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (RandomizedSearchCV,
                                     train_test_split)
from sklearn.metrics import (classification_report, jaccard_score,
                             hamming_loss, log_loss, balanced_accuracy_score,
                             confusion_matrix, r2_score, mean_absolute_error,
                             mean_squared_error, explained_variance_score)

from business_checkins import load_dataframe_from_yelp_2
from model_setup import ModelSetupInfo


class ModelDetailsStorage():
    """
    Storage class for keeping track of the inputs and results of
    the modeling process.
    """
    def __init__(self):
        self.path_to_records_file = \
            '~/Documents/Galvanize_DSI/' \
            'capstones/C2_Yelp_Review_Quality/models/'
        self.full_records = \
            pd.read_csv(self.path_to_records_file + 'model_info.csv')
        self.working_records_list = []
        self.working_record = {}
        self.template_record = \
            {'record_id': None, 'CV_fit_time': None, 'CV_score_time': None,
             'refit_time': None, 'CV_accuracy': None,
             'CV_balanced_accuracy': None, 'CV_F1_score': None,
             'CV_precision': None, 'CV_recall': None,
             'CV_roc_auc': None,  'CV_R2_score': None,
             'CV_mse': None, 'CV_rmse': None, 'CV_mae': None,
             'Test_accuracy': None, 'Test_balanced_accuracy': None,
             'Test_f1_score': None, 'Test_precision': None,
             'Test_recall': None, 'Test_hamming_loss': None,
             'Test_jaccard_score': None, 'Test_log_loss': None,
             'Test_classification_report': None,
             'Test_false_negatives': None, 'Test_false_positives': None,
             'Test_true_negatives': None, 'Test_true_positives': None,
             'Test_confusion_matrix': None, 'Test_r2_score': None,
             'Test_mse': None, 'Test_rmse': None, 'Test_mae': None,
             'Test_explained_variance_score': None, 'data': None,
             'goal': None,  'model_type': None, 'question': None,
             'record_count': None, 'target': None, 'scalar': None,
             'hyperparameters': None, 'features': None}

    def save_to_file(self):
        self.full_records.to_csv(self.path_to_records_file + 'model_info.csv',
                                 index=False)

    def add_working_list_to_full_records(self):
        data_to_add = self.working_records_list.copy()
        data_to_add = pd.DataFrame.from_records(data_to_add)
        self.full_records = self.full_records.append(data_to_add,
                                                     ignore_index=True)

    def add_record_to_list(self):
        copy = self.working_record.copy()
        copy['record_id'] = \
            ''.join([random.choice(string.ascii_letters
                                   + string.digits) for n in range(10)])
        self.working_records_list.append(copy)

    def clear_working_records_list(self):
        self.working_records_list = []

    def clear_record(self):
        self.working_record = {}

    def reset_record(self):
        self.working_record = self.template_record

    def print_record(self):
        pprint.pprint(self.working_record)

    def print_list(self):
        pprint.pprint(self.working_records_list)

    def print_full_records_info(self):
        print(self.full_records.info())
        print(self.full_records.head(20))


class ModelPipeline():
    """
    Full model testing pipeline from loading data to prediction metrics.
    """
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.Model = None
        self.goal = None
        self.y_pred = None
        self.y_pred_proba = None

    def load_data(self, question, records):
        """
        Loads data from yelp_2 table on postgres,
        Updates model_details.working_record columns question and record_count.

        Args:
            question (str): Options: ('td', 'non_td')
                'td' for time discounted features.
                Answering the question:
                Review quality at time of review.
                'non_td' for non time discounted features.
                Answering the question:
                Review quality on "found" reviews at a later date.

            records (int): Number of records to load.

        Returns:
            Dataframe: Results of query.
        """
        table = None
        if question == 'td':
            table = 'working_td_data'
        elif question == 'non_td':
            table = 'working_non_td_data'
        else:
            print('Invalid question argument')
            exit()
        record_count = records
        query = f'''
                SELECT *
                FROM {table}
                LIMIT {record_count}
                ;
                '''
        df = load_dataframe_from_yelp_2(query)
        df = df.replace([np.inf, -np.inf], np.nan)
        nan_count = sum(df.isna().sum())
        if nan_count > 0:
            df = df.dropna()
            print('This data has nans. They are being dropped.')
        model_details.working_record['question'] = question
        model_details.working_record['record_count'] = len(df.index)
        self.data = df

    def prep_data(self, datatype, target, scalar):
        """
        Takes in dataframe. Selects appropriate columns.
        Scales, shuffles, stratifies, and splits the data.
        Updates model_details.working_record columns:
        date, target, features, goal, scalar.

        Args:
            datatype (str): Options: ('text', 'non_text', 'both)
                Whether to load the review text, the metadata, or both.

            target (str): Options: ('T1_REG_review_total_ufc',
                                    'T2_CLS_ufc_>0', 'T3_CLS_ufc_level',
                                    'T4_REG_ufc_TD', 'T5_CLS_ufc_level_TD',
                                    'T6_REG_ufc_TDBD')
                Name of target column.

            scalar (str): Options: ('standard', 'power', 'no_scaling')
                How to scale the data. Reccommend 'power' since
                a lot of data is heavily skewed.

        Returns:
            Tuple of Dataframes and Series: X_train, X_test, y_train, y_test
        """
        model_setup = ModelSetupInfo()
        data = None
        if datatype == 'text':
            data = self.data.loc[:, ['review_id', 'review_text',
                                     'T1_REG_review_total_ufc',
                                     'T2_CLS_ufc_>0', 'T3_CLS_ufc_level',
                                     'T4_REG_ufc_TD', 'T5_CLS_ufc_level_TD',
                                     'T6_REG_ufc_TDBD']]
        elif datatype == 'non_text':
            data = self.data.drop('review_text', axis=1)
        elif datatype == 'both':
            data = self.data
        else:
            print('Invalid datatype argument')
            exit()

        reg_targets = ['T1_REG_review_total_ufc', 'T4_REG_ufc_TD',
                       'T6_REG_ufc_TDBD']
        cls_targets = ['T2_CLS_ufc_>0', 'T3_CLS_ufc_level',
                       'T5_CLS_ufc_level_TD']
        if target in cls_targets:
            self.goal = 'cls'
        elif target in reg_targets:
            self.goal = 'reg'
        else:
            print('Invalid target argument')
            exit()

        if scalar == 'standard':
            data_scale = 'Standard'
        elif scalar == 'power':
            data_scale = 'Power'
        elif scalar == 'no_scaling':
            data_scale = None
        else:
            print('Invalid scalar argument')
            exit()

        target_data = data[target]
        non_features = reg_targets + cls_targets
        non_features.append('review_id')
        features_data = data.drop(labels=non_features, axis=1)
        model_details.working_record['data'] = datatype
        model_details.working_record['target'] = target
        model_details.working_record['features'] = list(features_data.columns)
        model_details.working_record['goal'] = self.goal
        model_details.working_record['scalar'] = scalar
        if data_scale is not None:
            features_data = \
                model_setup.scalars[data_scale]().fit_transform(features_data)

        if self.goal == 'cls':
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(features_data, target_data, test_size=0.20,
                                 random_state=7, shuffle=True,
                                 stratify=target_data)
        elif self.goal == 'reg':
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(features_data, target_data, test_size=0.20,
                                 random_state=7, shuffle=True)

    def fit_model_CV(self, model_type):
        """
        Performs cross validation and
        fits model to best param combo.
        Updates model_details.working_record column model_type.

        Args:
            model_type (str): Options: (Elastic Net, Forest Reg,
                                        HGB Reg, XGB Reg, Log Reg,
                                        Forest Cls, HGB Cls, XGB Cls)
                Which base model to use.

        Returns:
            Object: Fitted model.
        """
        model_setup = ModelSetupInfo()
        if (model_type in model_setup.cls_models) and (self.goal == 'cls'):
            ModelCV = RandomizedSearchCV(model_setup.cls_models[model_type](),
                                         model_setup.cls_params[model_type],
                                         n_iter=10,
                                         scoring=model_setup.cls_scoring,
                                         n_jobs=None,
                                         cv=None,
                                         refit='Accuracy',
                                         random_state=7)
        elif (model_type in model_setup.reg_models) and (self.goal == 'reg'):
            ModelCV = RandomizedSearchCV(model_setup.reg_models[model_type](),
                                         model_setup.reg_params[model_type],
                                         n_iter=10,
                                         scoring=model_setup.reg_scoring,
                                         n_jobs=None,
                                         cv=None,
                                         refit='R2 Score',
                                         random_state=7)
        else:
            print('''Invalid model type. Make sure the model is in model_setup
                     and has the correct goal.
                     (classification or regression)''')
            exit()
        ModelCV.fit(self.X_train, self.y_train)
        model_details.working_record['model_type'] = model_type
        self.Model = ModelCV

    def store_CV_metrics(self, record, cv_results, cv_results_index):
        """
        Helper functions for storing results of cross validation into
        model_details.working_record.

        Args:
            record (dict): Record to store values in.
            cv_results_index (int): Index in .cv_results_ to get data from.
            cv_results (Dataframe): .cv_results_ of fitted model after CV.
        """
        record['hyperparameters'] = \
            cv_results.loc[cv_results_index, 'params']
        record['CV_fit_time'] = \
            round(cv_results.loc[cv_results_index, 'mean_fit_time'], 5)
        record['CV_score_time'] = \
            round(cv_results.loc[cv_results_index, 'mean_score_time'], 5)
        if self.goal == 'cls':
            record['CV_accuracy'] = \
                round(cv_results.loc[cv_results_index,
                                     'mean_test_Accuracy'], 5)
            record['CV_balanced_accuracy'] = \
                round(cv_results.loc[cv_results_index,
                                     'mean_test_Balanced Accuracy'], 5)
            record['CV_precision'] = \
                round(cv_results.loc[cv_results_index,
                                     'mean_test_Precision'], 5)
            record['CV_recall'] = \
                round(cv_results.loc[cv_results_index, 'mean_test_Recall'], 5)
            record['CV_F1_score'] = \
                round(cv_results.loc[cv_results_index,
                                     'mean_test_F1_Score'], 5)
            record['CV_roc_auc'] = \
                round(cv_results.loc[cv_results_index, 'mean_test_ROC_AUC'], 5)
        elif self.goal == 'reg':
            record['CV_R2_score'] = \
                round(cv_results.loc[cv_results_index,
                                     'mean_test_R2 Score'], 5)
            record['CV_mse'] = \
                abs(round(cv_results.loc[cv_results_index,
                                         'mean_test_MSE'], 5))
            record['CV_rmse'] = \
                abs(round(cv_results.loc[cv_results_index,
                                         'mean_test_RMSE'], 5))
            record['CV_mae'] = \
                abs(round(cv_results.loc[cv_results_index,
                                         'mean_test_MAE'], 5))

    def store_CV_data(self):
        """
        Stores results of CV.
        """
        cv_results = pd.DataFrame(self.Model.cv_results_)
        for i in list(range(len(cv_results.index))):
            if i != self.Model.best_index_:
                self.store_CV_metrics(model_details.working_record,
                                      cv_results, i)
                model_details.add_record_to_list()
        self.store_CV_metrics(model_details.working_record, cv_results,
                              self.Model.best_index_)
        model_details.working_record['refit_time'] = \
            round(self.Model.refit_time_, 5)

    def predict_and_store(self):
        """
        Uses fitted model to predict on test set.
        Adds metrics to model_details.working_record.
        """
        self.y_pred = self.Model.predict(self.X_test)
        if self.goal == 'cls':
            self.y_pred_proba = \
                self.Model.predict_proba(self.X_test)
            report = classification_report(self.y_test, self.y_pred,
                                           output_dict=True)
            balanced_accuracy = round(balanced_accuracy_score(self.y_test,
                                                              self.y_pred), 5)
            jaccard = round(jaccard_score(self.y_test, self.y_pred,
                                          average='weighted'), 5)
            hamming = round(hamming_loss(self.y_test, self.y_pred), 5)
            log = round(log_loss(self.y_test, self.y_pred_proba), 5)
            accuracy = round(report['accuracy'], 5)
            precision = round(report['weighted avg']['precision'], 5)
            recall = round(report['weighted avg']['recall'], 5)
            f1_score = round(report['weighted avg']['f1-score'], 5)
            confusion_mtx = confusion_matrix(self.y_test, self.y_pred)
            if model_details.working_record['target'] == 'T2_CLS_ufc_>0':
                tn, fp, fn, tp = confusion_mtx.ravel()
                model_details.working_record['Test_true_negatives'] = tn
                model_details.working_record['Test_false_positives'] = fp
                model_details.working_record['Test_false_negatives'] = fn
                model_details.working_record['Test_true_positives'] = tp
            model_details.working_record['Test_classification_report'] = report
            model_details.working_record['Test_balanced_accuracy'] = \
                balanced_accuracy
            model_details.working_record['Test_jaccard_score'] = jaccard
            model_details.working_record['Test_hamming_loss'] = hamming
            model_details.working_record['Test_log_loss'] = log
            model_details.working_record['Test_accuracy'] = accuracy
            model_details.working_record['Test_precision'] = precision
            model_details.working_record['Test_recall'] = recall
            model_details.working_record['Test_f1_score'] = f1_score
            model_details.working_record['Test_confusion_matrix'] = \
                confusion_mtx
        elif self.goal == 'reg':
            R2_score = round(r2_score(self.y_test, self.y_pred), 5)
            mse = round(mean_squared_error(self.y_test, self.y_pred), 5)
            rmse = round(mean_squared_error(self.y_test, self.y_pred,
                                            squared=False), 5)
            mae = round(mean_absolute_error(self.y_test, self.y_pred), 5)
            explained_variance = \
                round(explained_variance_score(self.y_test, self.y_pred), 5)
            model_details.working_record['Test_r2_score'] = R2_score
            model_details.working_record['Test_mse'] = mse
            model_details.working_record['Test_rmse'] = rmse
            model_details.working_record['Test_mae'] = mae
            model_details.working_record['Test_explained_variance_score'] = \
                explained_variance
        model_details.add_record_to_list()

    def save_model_to_pickle(self):
        save_path = '~/Documents/Galvanize_DSI/' \
                    'capstones/C2_Yelp_Review_Quality/models'
        name_of_file = input("What is the name of the model: ")
        completeName = os.path.join(save_path, name_of_file+".pkl")
        pickle.dump(self.Model, open(completeName, 'wb'))

    def load_model_from_pickle(self):
        save_path = '~/Documents/Galvanize_DSI/' \
                    'capstones/C2_Yelp_Review_Quality/models'
        name_of_file = input("What is the name of the model: ")
        completeName = os.path.join(save_path, name_of_file+".pkl")
        self.Model = pickle.load(open(completeName, 'wb'))


if __name__ == "__main__":
    model_details = ModelDetailsStorage()

    pipeline = ModelPipeline()
    pipeline.load_data('non_td', 1000)
    pipeline.prep_data('non_text', 'T2_CLS_ufc_>0', 'power')
    pipeline.fit_model_CV('HGB Cls')
    pipeline.store_CV_data()
    pipeline.predict_and_store()

    model_details.print_record()
    # model_details.add_working_list_to_full_records()
    # model_details.print_full_records_info()
    # model_details.save_to_file()
