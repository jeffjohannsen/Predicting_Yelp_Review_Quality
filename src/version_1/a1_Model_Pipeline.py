import random
import string
import pprint
import pickle
import time
import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    classification_report,
    jaccard_score,
    hamming_loss,
    log_loss,
    balanced_accuracy_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
)
from imblearn.over_sampling import SMOTE

from business_checkins import load_dataframe_from_yelp_2
from a4_Model_Setup import ModelSetupInfo
from a2_NLP import NLPPipeline

pd.set_option("display.max_columns", 100)


class ModelDetailsStorage:
    """
    Storage class for keeping track of the inputs and results of
    the modeling process.
    """

    def __init__(self, run_on_aws):
        self.aws = run_on_aws
        if self.aws:
            self.path_to_records_file = (
                "~/Predicting-Yelp-Review-Quality/models/"
            )
        else:
            self.path_to_records_file = (
                "~/Documents/Galvanize_DSI/"
                "capstones/C2_Yelp_Review_Quality/models/"
            )
        self.full_records = pd.read_csv(
            self.path_to_records_file + "model_info.csv"
        )
        self.working_records_list = []
        self.template_record = {
            "record_id": None,
            "record_type": None,
            "CV_fit_time": None,
            "CV_score_time": None,
            "refit_time": None,
            "CV_accuracy": None,
            "CV_balanced_accuracy": None,
            "CV_F1_score": None,
            "CV_precision": None,
            "CV_recall": None,
            "CV_roc_auc": None,
            "CV_R2_score": None,
            "CV_mse": None,
            "CV_rmse": None,
            "CV_mae": None,
            "Test_accuracy": None,
            "Test_balanced_accuracy": None,
            "Test_f1_score": None,
            "Test_precision": None,
            "Test_recall": None,
            "Test_hamming_loss": None,
            "Test_jaccard_score": None,
            "Test_log_loss": None,
            "Test_classification_report": None,
            "Test_false_negatives": None,
            "Test_false_positives": None,
            "Test_true_negatives": None,
            "Test_true_positives": None,
            "Test_confusion_matrix": None,
            "Test_r2_score": None,
            "Test_mse": None,
            "Test_rmse": None,
            "Test_mae": None,
            "Test_explained_variance_score": None,
            "data": None,
            "goal": None,
            "model_type": None,
            "question": None,
            "record_count": None,
            "target": None,
            "scalar": None,
            "balancer": None,
            "hyperparameters": None,
            "features": None,
            "full_runtime_minutes": None,
        }
        self.working_record = self.template_record.copy()

    def save_to_file(self):
        self.full_records.to_csv(
            self.path_to_records_file + "model_info.csv", index=False
        )

    def add_working_list_to_full_records(self):
        data_to_add = self.working_records_list.copy()
        data_to_add = pd.DataFrame.from_records(data_to_add)
        self.full_records = self.full_records.append(
            data_to_add, ignore_index=True
        )

    def add_record_to_list(self):
        copy = self.working_record.copy()
        copy["record_id"] = "".join(
            [
                random.choice(string.ascii_letters + string.digits)
                for n in range(10)
            ]
        )
        self.working_records_list.append(copy)

    def clear_working_records_list(self):
        self.working_records_list = []

    def clear_record(self):
        self.working_record = {}

    def reset_record(self):
        self.working_record = self.template_record.copy()

    def print_record(self):
        pprint.pprint(self.working_record)

    def print_list(self):
        records = self.working_records_list.copy()
        records = pd.DataFrame.from_records(records)
        records_to_print = records[records["record_type"] == "test"]
        columns_to_print = [
            "record_count",
            "question",
            "data",
            "goal",
            "target",
            "model_type",
            "Test_accuracy",
            "Test_f1_score",
            "Test_r2_score",
            "Test_rmse",
        ]
        records_to_print = records_to_print.loc[:, columns_to_print]
        print(records_to_print)
        # pprint.pprint(self.working_records_list)

    def reset_full_records(self):
        self.full_records = pd.DataFrame(self.template_record, index=[0])

    def print_full_records_info(self):
        print(self.full_records.info())
        print(self.full_records.head(20))


class ModelPipeline:
    """
    Full model testing pipeline from loading data to prediction metrics.
    """

    def __init__(self, run_on_aws=False):
        """
        Args:
            run_on_aws (bool, optional): True if running on aws, else False.
                                         Defaults to False.
        """
        self.aws = run_on_aws
        self.setup = ModelSetupInfo()
        self.model_details = ModelDetailsStorage(self.aws)
        self.use_cv = None
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.Model = None
        self.goal = None
        self.datatype = None

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
        if question == "td":
            table = "working_td_data"
        elif question == "non_td":
            table = "working_non_td_data"
        else:
            print("Invalid question argument")
            exit()
        record_count = records
        query = f"""
                SELECT *
                FROM {table}
                LIMIT {record_count}
                ;
                """
        df = load_dataframe_from_yelp_2(query)
        df = df.replace([np.inf, -np.inf], np.nan)
        nan_count = sum(df.isna().sum())
        if nan_count > 0:
            df = df.dropna()
            print(f"This data has {nan_count} nans. They are being dropped.")
        self.model_details.working_record["question"] = question
        self.model_details.working_record["record_count"] = len(df.index)
        self.data = df

    def prep_data(self, datatype, target):
        """
        Takes in dataframe. Selects appropriate columns.
        Scales, shuffles, stratifies, and splits the data.
        Updates model_details.working_record columns:
        date, target, features, goal, scalar.

        Args:
            datatype (str): Options: ('text', 'non_text', 'both')
                Whether to load the review text, the metadata, or both.

            target (str): Options: ('T1_REG_review_total_ufc',
                                    'T2_CLS_ufc_>0', 'T3_CLS_ufc_level',
                                    'T4_REG_ufc_TD', 'T5_CLS_ufc_level_TD',
                                    'T6_REG_ufc_TDBD')
                Name of target column.

        Returns:
            Tuple of Dataframes and Series: X_train, X_test, y_train, y_test
        """
        if datatype == "text":
            data = self.data.loc[
                :,
                [
                    "review_id",
                    "review_text",
                    "T1_REG_review_total_ufc",
                    "T2_CLS_ufc_>0",
                    "T3_CLS_ufc_level",
                    "T4_REG_ufc_TD",
                    "T5_CLS_ufc_level_TD",
                    "T6_REG_ufc_TDBD",
                ],
            ]
        elif datatype == "non_text":
            data = self.data.drop("review_text", axis=1)
        elif datatype == "both":
            data = self.data
        else:
            print("Invalid datatype argument")
            exit()
        self.datatype = datatype

        reg_targets = [
            "T1_REG_review_total_ufc",
            "T4_REG_ufc_TD",
            "T6_REG_ufc_TDBD",
        ]
        cls_targets = [
            "T2_CLS_ufc_>0",
            "T3_CLS_ufc_level",
            "T5_CLS_ufc_level_TD",
        ]
        if target in cls_targets:
            self.goal = "cls"
        elif target in reg_targets:
            self.goal = "reg"
        else:
            print("Invalid target argument")
            exit()

        target_data = data[target]
        non_features = reg_targets + cls_targets
        features_data = data.drop(labels=non_features, axis=1)
        self.model_details.working_record["target"] = target
        self.model_details.working_record["goal"] = self.goal

        if self.goal == "cls":
            (
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
            ) = train_test_split(
                features_data,
                target_data,
                test_size=0.20,
                random_state=7,
                shuffle=True,
                stratify=target_data,
            )
        elif self.goal == "reg":
            (
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
            ) = train_test_split(
                features_data,
                target_data,
                test_size=0.20,
                random_state=7,
                shuffle=True,
            )

    def add_nlp_features(
        self, show_model_results=False, use_cv=False, feature_level=1
    ):
        """
        Calls NLPPipeline to add text features to self.X_train self.and X_test.

        Args:
            show_model_results (bool, optional): Defaults to False.
                Whether to print metrics of nlp modeling.

            use_cv (bool, optional): Defaults to False.
                Whether to use cross validation in training nlp models.

            feature_level (int, optional): Options: (1,2,3,4) Defaults to 1.
                Number of features to add.
                Increase in levels significantly increases compute time.
                Level 1: Basic Text Counts
                Level 2: Model Predictions
                Level 3: Spacy Language Features
                Level 4: Text Readability Grade Level
        """
        self.use_cv = use_cv
        if self.datatype != "non_text":
            training_text_features = self.X_train.loc[
                :, ["review_id", "review_text"]
            ].copy()
            testing_text_features = self.X_test.loc[
                :, ["review_id", "review_text"]
            ].copy()
            nlp_pipeline = NLPPipeline(
                goal=self.goal,
                X_train=training_text_features,
                y_train=self.y_train.copy(),
                X_test=testing_text_features,
                y_test=self.y_test.copy(),
                show_results=show_model_results,
                use_cv=use_cv,
            )

            nlp_pipeline.create_basic_text_features()
            if feature_level >= 2:
                nlp_pipeline.create_prediction_features()
            if feature_level >= 3:
                nlp_pipeline.create_spacy_features()
            if feature_level >= 4:
                nlp_pipeline.create_readability_features()

            nlp_pipeline.X_train.drop("review_text", axis=1, inplace=True)
            nlp_pipeline.X_test.drop("review_text", axis=1, inplace=True)
            nlp_pipeline.X_train.rename(
                columns={"review_id": "nlp_review_id"}, inplace=True
            )
            nlp_pipeline.X_test.rename(
                columns={"review_id": "nlp_review_id"}, inplace=True
            )
            self.X_train = pd.concat(
                [self.X_train, nlp_pipeline.X_train], axis=1
            )
            self.X_test = pd.concat([self.X_test, nlp_pipeline.X_test], axis=1)

            mismatched_id_count = len(
                self.X_train[
                    self.X_train["review_id"] != self.X_train["nlp_review_id"]
                ]
            )
            if mismatched_id_count > 0:
                print(
                    f"Error Combining NLP and Main data. "
                    "Mismatched ids: {mismatched_id_count}"
                )
                exit()
            self.X_train.drop(
                labels=(["nlp_review_id", "review_text"]), axis=1, inplace=True
            )
            self.X_test.drop(
                labels=(["nlp_review_id", "review_text"]), axis=1, inplace=True
            )

        self.X_train.drop("review_id", axis=1, inplace=True)
        self.X_test.drop("review_id", axis=1, inplace=True)
        self.model_details.working_record["data"] = self.datatype
        self.model_details.working_record["features"] = list(
            self.X_train.columns
        )

    def fit_model(self, use_cv, model_type, scalar, balancer=None):
        """
        Performs cross validation and
        fits model to best param combo.
        Updates model_details.working_record column model_type.

        Args:
            use_cv (bool): Whether to use cross validation.

            model_type (str): Options: (Elastic Net, Forest Reg,
                                        HGB Reg, XGB Reg, Log Reg,
                                        Forest Cls, HGB Cls, XGB Cls)
                Which base model to use.

            scalar (str): Options: ('standard', 'power', 'no_scaling')
                How to scale the data. Reccommend 'power' since
                a lot of data is heavily skewed.

            balancer (str, None): Options: ('smote') Defaults to None.
                How to balance uneven class weights if goal is 'cls'.

        Returns:
            Object: Fitted model.
        """
        self.use_cv = use_cv
        feature_labels = self.X_train.columns

        if scalar == "standard":
            data_scale = "Standard"
        elif scalar == "power":
            data_scale = "Power"
        elif scalar == "no_scaling":
            data_scale = None
        else:
            print("Invalid scalar argument")
            exit()
        self.model_details.working_record["scalar"] = scalar
        if data_scale is not None:
            scalar = self.setup.scalars[data_scale]()
            self.X_train = scalar.fit_transform(self.X_train)
            self.X_test = scalar.transform(self.X_test)
            self.X_train = pd.DataFrame(self.X_train, columns=feature_labels)
            self.X_test = pd.DataFrame(self.X_test, columns=feature_labels)

        if balancer not in [None, "smote"]:
            print("Invalid balancer argument")
            exit()
        self.model_details.working_record["balancer"] = balancer
        if (self.goal == "cls") and (balancer == "smote"):
            smote = SMOTE(random_state=7)
            self.X_train, self.y_train = smote.fit_resample(
                self.X_train, self.y_train
            )

        if (model_type in self.setup.cls_models) and (self.goal == "cls"):
            if use_cv:
                param_grid = self.setup.cls_params_cv[model_type]
                Model = GridSearchCV(
                    self.setup.cls_models[model_type](),
                    param_grid,
                    scoring=self.setup.cls_scoring,
                    n_jobs=-1,
                    cv=None,
                    refit="Accuracy",
                )
            else:
                params = self.setup.cls_params[model_type]
                Model = self.setup.cls_models[model_type](**params)
        elif (model_type in self.setup.reg_models) and (self.goal == "reg"):
            if use_cv:
                param_grid = self.setup.reg_params_cv[model_type]
                Model = GridSearchCV(
                    self.setup.reg_models[model_type](),
                    param_grid,
                    scoring=self.setup.reg_scoring,
                    n_jobs=-1,
                    cv=None,
                    refit="R2 Score",
                )
            else:
                params = self.setup.reg_params[model_type]
                Model = self.setup.reg_models[model_type](**params)
        else:
            print(
                "Invalid model type. Make sure the model is in model_setup "
                "and has the correct goal. "
                "(classification or regression)"
            )
            exit()
        Model.fit(self.X_train, self.y_train)
        self.Model = Model
        self.model_details.working_record["balancer"] = balancer
        self.model_details.working_record["model_type"] = model_type

    def store_CV_metrics(self, record, cv_results, cv_results_index):
        """
        Helper functions for storing results of cross validation into
        model_details.working_record.

        Args:
            record (dict): Record to store values in.
            cv_results_index (int): Index in .cv_results_ to get data from.
            cv_results (Dataframe): .cv_results_ of fitted model after CV.
        """
        record["hyperparameters"] = cv_results.loc[cv_results_index, "params"]
        record["CV_fit_time"] = round(
            cv_results.loc[cv_results_index, "mean_fit_time"], 5
        )
        record["CV_score_time"] = round(
            cv_results.loc[cv_results_index, "mean_score_time"], 5
        )
        if self.goal == "cls":
            record["CV_accuracy"] = round(
                cv_results.loc[cv_results_index, "mean_test_Accuracy"], 5
            )
            record["CV_balanced_accuracy"] = round(
                cv_results.loc[
                    cv_results_index, "mean_test_Balanced Accuracy"
                ],
                5,
            )
            record["CV_precision"] = round(
                cv_results.loc[cv_results_index, "mean_test_Precision"], 5
            )
            record["CV_recall"] = round(
                cv_results.loc[cv_results_index, "mean_test_Recall"], 5
            )
            record["CV_F1_score"] = round(
                cv_results.loc[cv_results_index, "mean_test_F1_Score"], 5
            )
            record["CV_roc_auc"] = round(
                cv_results.loc[cv_results_index, "mean_test_ROC_AUC"], 5
            )
        elif self.goal == "reg":
            record["CV_R2_score"] = round(
                cv_results.loc[cv_results_index, "mean_test_R2 Score"], 5
            )
            record["CV_mse"] = abs(
                round(cv_results.loc[cv_results_index, "mean_test_MSE"], 5)
            )
            record["CV_rmse"] = abs(
                round(cv_results.loc[cv_results_index, "mean_test_RMSE"], 5)
            )
            record["CV_mae"] = abs(
                round(cv_results.loc[cv_results_index, "mean_test_MAE"], 5)
            )

    def store_CV_data(self):
        """
        Stores results of CV.
        """
        if self.use_cv:
            cv_results = pd.DataFrame(self.Model.cv_results_)
            for i in list(range(len(cv_results.index))):
                if i != self.Model.best_index_:
                    self.store_CV_metrics(
                        self.model_details.working_record, cv_results, i
                    )
                    self.model_details.working_record["record_type"] = "cv"
                    self.model_details.add_record_to_list()
            self.store_CV_metrics(
                self.model_details.working_record,
                cv_results,
                self.Model.best_index_,
            )
            self.model_details.working_record["refit_time"] = round(
                self.Model.refit_time_, 5
            )

    def predict_and_store(self):
        """
        Uses fitted model to predict on test set.
        Adds metrics to model_details.working_record.
        """
        self.model_details.working_record["record_type"] = "test"
        self.y_pred = self.Model.predict(self.X_test)
        if self.goal == "cls":
            self.y_pred_proba = self.Model.predict_proba(self.X_test)
            report = classification_report(
                self.y_test, self.y_pred, output_dict=True
            )
            balanced_accuracy = round(
                balanced_accuracy_score(self.y_test, self.y_pred), 5
            )
            jaccard = round(
                jaccard_score(self.y_test, self.y_pred, average="weighted"), 5
            )
            hamming = round(hamming_loss(self.y_test, self.y_pred), 5)
            log = round(log_loss(self.y_test, self.y_pred_proba), 5)
            accuracy = round(report["accuracy"], 5)
            precision = round(report["weighted avg"]["precision"], 5)
            recall = round(report["weighted avg"]["recall"], 5)
            f1_score = round(report["weighted avg"]["f1-score"], 5)
            confusion_mtx = confusion_matrix(self.y_test, self.y_pred)
            if self.model_details.working_record["target"] == "T2_CLS_ufc_>0":
                tn, fp, fn, tp = confusion_mtx.ravel()
                self.model_details.working_record["Test_true_negatives"] = tn
                self.model_details.working_record["Test_false_positives"] = fp
                self.model_details.working_record["Test_false_negatives"] = fn
                self.model_details.working_record["Test_true_positives"] = tp
            self.model_details.working_record[
                "Test_classification_report"
            ] = report
            self.model_details.working_record[
                "Test_balanced_accuracy"
            ] = balanced_accuracy
            self.model_details.working_record["Test_jaccard_score"] = jaccard
            self.model_details.working_record["Test_hamming_loss"] = hamming
            self.model_details.working_record["Test_log_loss"] = log
            self.model_details.working_record["Test_accuracy"] = accuracy
            self.model_details.working_record["Test_precision"] = precision
            self.model_details.working_record["Test_recall"] = recall
            self.model_details.working_record["Test_f1_score"] = f1_score
            self.model_details.working_record[
                "Test_confusion_matrix"
            ] = confusion_mtx
        elif self.goal == "reg":
            R2_score = round(r2_score(self.y_test, self.y_pred), 5)
            mse = round(mean_squared_error(self.y_test, self.y_pred), 5)
            rmse = round(
                mean_squared_error(self.y_test, self.y_pred, squared=False), 5
            )
            mae = round(mean_absolute_error(self.y_test, self.y_pred), 5)
            explained_variance = round(
                explained_variance_score(self.y_test, self.y_pred), 5
            )
            self.model_details.working_record["Test_r2_score"] = R2_score
            self.model_details.working_record["Test_mse"] = mse
            self.model_details.working_record["Test_rmse"] = rmse
            self.model_details.working_record["Test_mae"] = mae
            self.model_details.working_record[
                "Test_explained_variance_score"
            ] = explained_variance

    def save_model_to_pickle(self):
        save_path = (
            "~/Documents/Galvanize_DSI/"
            "capstones/C2_Yelp_Review_Quality/models"
        )
        name_of_file = input("What is the name of the model: ")
        completeName = os.path.join(save_path, name_of_file + ".pkl")
        pickle.dump(self.Model, open(completeName, "wb"))

    def load_model_from_pickle(self):
        save_path = (
            "~/Documents/Galvanize_DSI/"
            "capstones/C2_Yelp_Review_Quality/models"
        )
        name_of_file = input("What is the name of the model: ")
        completeName = os.path.join(save_path, name_of_file + ".pkl")
        self.Model = pickle.load(open(completeName, "wb"))

    def run_full_pipeline(
        self,
        use_cv,
        print_results,
        save_results,
        question,
        records,
        data,
        target,
        model,
        scalar="power",
        balancer="smote",
    ):
        """
        Wrapper function for the full model pipeline.
        Arguments will be partially validated.
        See model_setup.py for full options list.

        Args:
            records (int): Record count to load.
                Larger numbers may overload memory
                and/or dramtically increase computation time.
                Max records avaliable: 6241598.

            use_cv (bool): Whether to use cross validation.
                Increases computation time.

            print_results (bool): Whether to print the modeling results.
            save_results (bool): Whether to save the modeling results.
            question (str):
            data (str):
            target (str):
            model (str):
            scalar (str, optional): Defaults to 'power'.
            balancer (str, optional): Defaults to 'smote'.
        """
        sta = time.perf_counter()
        param_options = {
            "question": ["td", "non_td"],
            "data": ["text", "non_text", "both"],
            "target": [
                "T1_REG_review_total_ufc",
                "T2_CLS_ufc_>0",
                "T3_CLS_ufc_level",
                "T4_REG_ufc_TD",
                "T5_CLS_ufc_level_TD",
                "T6_REG_ufc_TDBD",
            ],
            "model": [
                "Log Reg",
                "Forest Cls",
                "HGB Cls",
                "XGB Cls",
                "Elastic Net",
                "Forest Reg",
                "HGB Reg",
                "XGB Reg",
            ],
            "scalar": ["power", "standard", "no_scaling"],
            "balancer": ["smote", None],
        }

        for k, v in {
            "question": question,
            "data": data,
            "target": target,
            "model": model,
            "scalar": scalar,
            "balancer": balancer,
        }.items():
            validate_input(k, v, param_options[k])

        st = time.perf_counter()
        self.load_data(question, records)
        ft = time.perf_counter()
        print(f"Loaded Data in {(ft - st):.4f} seconds")
        st = time.perf_counter()
        self.prep_data(data, target)
        ft = time.perf_counter()
        print(f"Prepped Data in {(ft - st):.4f} seconds")
        st = time.perf_counter()
        self.add_nlp_features(
            show_model_results=print_results, use_cv=use_cv, feature_level=4
        )
        ft = time.perf_counter()
        print(f"Added NLP Features in {((ft - st) / 60):.4f} minutes")
        st = time.perf_counter()
        self.fit_model(use_cv, model, scalar, balancer)
        ft = time.perf_counter()
        print(f"Fitted Main Model and CV in {((ft - st) / 60):.4f} minutes")
        st = time.perf_counter()
        self.store_CV_data()
        ft = time.perf_counter()
        print(f"Stored CV data in {(ft - st):.4f} seconds")
        st = time.perf_counter()
        self.predict_and_store()
        ft = time.perf_counter()
        print(f"Predicted and Evaluated in {((ft - st) / 60):.4f} minutes")
        fta = time.perf_counter()
        print(f"Full Pipeline Run in {((fta - sta) / 60):.4f} minutes")
        self.model_details.working_record["full_runtime_minutes"] = (
            fta - sta
        ) / 60
        self.model_details.add_record_to_list()
        if print_results:
            self.model_details.print_list()
        self.model_details.add_working_list_to_full_records()
        if save_results:
            self.model_details.save_to_file()
        self.model_details.reset_record()
        self.model_details.clear_working_records_list()
        return (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.Model,
        )


def validate_input(argument, input_value, options):
    if input_value not in options:
        print(f"Invalid {argument} argument.")
        print(f"Available options include: {options}")
        exit()


if __name__ == "__main__":
    # run_full_pipeline(use_cv=True, print_results=True, save_results=False,
    #                   question='non_td', records=1000, data='both',
    #                   target='T2_CLS_ufc_>0', model='Log Reg',
    #                   scalar='power', balancer='smote')
    for data in ["text", "non_text", "both"]:
        for target in ["T2_CLS_ufc_>0", "T5_CLS_ufc_level_TD"]:
            for model in ["Log Reg"]:
                pipeline = ModelPipeline(run_on_aws=False)
                pipeline.run_full_pipeline(
                    use_cv=True,
                    print_results=True,
                    save_results=True,
                    question="td",
                    records=10000,
                    data=data,
                    target=target,
                    model=model,
                    scalar="power",
                    balancer="smote",
                )
