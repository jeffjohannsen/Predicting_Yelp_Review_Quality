# Imports and Global Settings
import time
import mlflow
import numpy as np
import pandas as pd
import pickle as pkl
from joblib import dump, load
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, roc_auc_score

pd.set_option("display.float_format", lambda x: "%.5f" % x)
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)

mlflow.sklearn.autolog()


class Base_Model_Process:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.model_predictions = None
        self.model_predict_proba = None

    def fit_model(self, X_train, y_train):
        start = time.perf_counter()
        with mlflow.start_run() as run:
            self.model.fit(X_train, y_train)
        end = time.perf_counter()
        print(f"\n{self.model_name} Fit Complete")
        print(f"Training took {(end-start)/60:.2f} minutes.")

    def predict_model(self, X_test):
        start = time.perf_counter()
        self.model_predictions = self.model.predict(X_test)
        self.model_predict_proba = self.model.predict_proba(X_test)[:, 1]
        end = time.perf_counter()
        print(f"\n{self.model_name} Predictions Complete")
        print(f"Predictions took {(end-start)/60:.2f} minutes.")

    def score_model(self, y_test, create_log=False):
        start = time.perf_counter()
        cls_report = classification_report(y_test, self.model_predictions)
        auc_score = roc_auc_score(y_test, self.model_predict_proba)
        end = time.perf_counter()
        print(f"\n{self.model_name} Scoring Complete")
        print(f"Scoring took {(end-start):.2f} seconds.")
        print(f">>>>>AUC Score: {auc_score:.2f}<<<<<")
        print("----------------Classification Report----------------")
        print(cls_report)

    def save_model(self, model_filename, model_name_postfix):
        dump(self.model, f"{model_filename}{model_name_postfix}.joblib")
        print(f"{self.model_name} Save Complete")
        print("===========================================")


# Load Data
def load_data(
    ec2_or_local="local", train_rec_count=10000, test_rec_count=1000
):
    """Load train and test data from csv.

    Args:
        ec2_or_local (str, optional): Where program is being run. Defaults to "local".
        train_rec_count (int, optional): Max records: 5523992. Defaults to 10000.
        test_rec_count (int, optional): Max records: 1382379. Defaults to 1000.

    Returns:
        tuple: train and test datasets
    """
    start = time.perf_counter()
    data_location_ec2 = "/home/ubuntu/"
    data_location_local = "/home/jeff/Documents/Data_Science_Projects/Yelp_Reviews/data/full_data/model_ready/"
    if ec2_or_local == "local":
        filepath_prefix = data_location_local
    else:
        filepath_prefix = data_location_ec2

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
        "pos_adj_pct": "float32",
        "pos_adj_cnt": "int16",
        "pos_adp_pct": "float32",
        "pos_adp_cnt": "int16",
        "pos_adv_pct": "float32",
        "pos_adv_cnt": "int16",
        "pos_aux_pct": "float32",
        "pos_aux_cnt": "int16",
        "pos_conj_pct": "float32",
        "pos_conj_cnt": "int16",
        "pos_det_pct": "float32",
        "pos_det_cnt": "int16",
        "pos_intj_pct": "float32",
        "pos_intj_cnt": "int16",
        "pos_noun_pct": "float32",
        "pos_noun_cnt": "int16",
        "pos_num_pct": "float32",
        "pos_num_cnt": "int16",
        "pos_part_pct": "float32",
        "pos_part_cnt": "int16",
        "pos_pron_pct": "float32",
        "pos_pron_cnt": "int16",
        "pos_propn_pct": "float32",
        "pos_propn_cnt": "int16",
        "pos_punct_pct": "float32",
        "pos_punct_cnt": "int16",
        "pos_sconj_pct": "float32",
        "pos_sconj_cnt": "int16",
        "pos_sym_pct": "float32",
        "pos_sym_cnt": "int16",
        "pos_verb_pct": "float32",
        "pos_verb_cnt": "int16",
        "pos_x_pct": "float32",
        "pos_x_cnt": "int16",
        "dep_root_pct": "float32",
        "dep_root_cnt": "int16",
        "dep_acl_pct": "float32",
        "dep_acl_cnt": "int16",
        "dep_acomp_pct": "float32",
        "dep_acomp_cnt": "int16",
        "dep_advcl_pct": "float32",
        "dep_advcl_cnt": "int16",
        "dep_advmod_pct": "float32",
        "dep_advmod_cnt": "int16",
        "dep_agent_pct": "float32",
        "dep_agent_cnt": "int16",
        "dep_amod_pct": "float32",
        "dep_amod_cnt": "int16",
        "dep_appos_pct": "float32",
        "dep_appos_cnt": "int16",
        "dep_attr_pct": "float32",
        "dep_attr_cnt": "int16",
        "dep_aux_pct": "float32",
        "dep_aux_cnt": "int16",
        "dep_auxpass_pct": "float32",
        "dep_auxpass_cnt": "int16",
        "dep_case_pct": "float32",
        "dep_case_cnt": "int16",
        "dep_cc_pct": "float32",
        "dep_cc_cnt": "int16",
        "dep_ccomp_pct": "float32",
        "dep_ccomp_cnt": "int16",
        "dep_compound_pct": "float32",
        "dep_compound_cnt": "int16",
        "dep_conj_pct": "float32",
        "dep_conj_cnt": "int16",
        "dep_csubj_pct": "float32",
        "dep_csubj_cnt": "int16",
        "dep_csubjpass_pct": "float32",
        "dep_csubjpass_cnt": "int16",
        "dep_dative_pct": "float32",
        "dep_dative_cnt": "int16",
        "dep_dep_pct": "float32",
        "dep_dep_cnt": "int16",
        "dep_det_pct": "float32",
        "dep_det_cnt": "int16",
        "dep_dobj_pct": "float32",
        "dep_dobj_cnt": "int16",
        "dep_expl_pct": "float32",
        "dep_expl_cnt": "int16",
        "dep_intj_pct": "float32",
        "dep_intj_cnt": "int16",
        "dep_mark_pct": "float32",
        "dep_mark_cnt": "int16",
        "dep_meta_pct": "float32",
        "dep_meta_cnt": "int16",
        "dep_neg_pct": "float32",
        "dep_neg_cnt": "int16",
        "dep_nmod_pct": "float32",
        "dep_nmod_cnt": "int16",
        "dep_npadvmod_pct": "float32",
        "dep_npadvmod_cnt": "int16",
        "dep_nsubj_pct": "float32",
        "dep_nsubj_cnt": "int16",
        "dep_nsubjpass_pct": "float32",
        "dep_nsubjpass_cnt": "int16",
        "dep_nummod_pct": "float32",
        "dep_nummod_cnt": "int16",
        "dep_oprd_pct": "float32",
        "dep_oprd_cnt": "int16",
        "dep_parataxis_pct": "float32",
        "dep_parataxis_cnt": "int16",
        "dep_pcomp_pct": "float32",
        "dep_pcomp_cnt": "int16",
        "dep_pobj_pct": "float32",
        "dep_pobj_cnt": "int16",
        "dep_poss_pct": "float32",
        "dep_poss_cnt": "int16",
        "dep_preconj_pct": "float32",
        "dep_preconj_cnt": "int16",
        "dep_predet_pct": "float32",
        "dep_predet_cnt": "int16",
        "dep_prep_pct": "float32",
        "dep_prep_cnt": "int16",
        "dep_prt_pct": "float32",
        "dep_prt_cnt": "int16",
        "dep_punct_pct": "float32",
        "dep_punct_cnt": "int16",
        "dep_quantmod_pct": "float32",
        "dep_quantmod_cnt": "int16",
        "dep_relcl_pct": "float32",
        "dep_relcl_cnt": "int16",
        "dep_xcomp_pct": "float32",
        "dep_xcomp_cnt": "int16",
        "ent_cardinal_pct": "float32",
        "ent_cardinal_cnt": "int16",
        "ent_date_pct": "float32",
        "ent_date_cnt": "int16",
        "ent_event_pct": "float32",
        "ent_event_cnt": "int16",
        "ent_fac_pct": "float32",
        "ent_fac_cnt": "int16",
        "ent_gpe_pct": "float32",
        "ent_gpe_cnt": "int16",
        "ent_language_pct": "float32",
        "ent_language_cnt": "int16",
        "ent_law_pct": "float32",
        "ent_law_cnt": "int16",
        "ent_loc_pct": "float32",
        "ent_loc_cnt": "int16",
        "ent_money_pct": "float32",
        "ent_money_cnt": "int16",
        "ent_norp_pct": "float32",
        "ent_norp_cnt": "int16",
        "ent_ordinal_pct": "float32",
        "ent_ordinal_cnt": "int16",
        "ent_org_pct": "float32",
        "ent_org_cnt": "int16",
        "ent_percent_pct": "float32",
        "ent_percent_cnt": "int16",
        "ent_person_pct": "float32",
        "ent_person_cnt": "int16",
        "ent_product_pct": "float32",
        "ent_product_cnt": "int16",
        "ent_quantity_pct": "float32",
        "ent_quantity_cnt": "int16",
        "ent_time_pct": "float32",
        "ent_time_cnt": "int16",
        "ent_work_of_art_pct": "float32",
        "ent_work_of_art_cnt": "int16",
    }

    train = pd.read_csv(
        f"{filepath_prefix}train.csv",
        nrows=train_rec_count,
        true_values=["True"],
        false_values=["False"],
        dtype=datatypes,
    )
    test = pd.read_csv(
        f"{filepath_prefix}test.csv",
        nrows=test_rec_count,
        true_values=["True"],
        false_values=["False"],
        dtype=datatypes,
    )

    end = time.perf_counter()

    print("\nData Load Complete")
    print(f"Took {(end-start):.2f} seconds")
    print(f"Train Shape: {train.shape}")
    print(f"Test Shape: {test.shape}")

    return train, test


def prep_data(
    train,
    test,
    feature_groups,
    scale_data=False,
):
    """Feature selection and data preprocessing.

    Args:
        feature_groups (string): Which features to use.
                                 Options:
                                 "all" - All features
                                 "submodels" - Only features from submodels
                                 "basic_text" - Only basic text features like word counts, etc.
                                 "non_linguistic" - All main features not including pos,dep,ent features from spacy
                                 "other" - review_stars, grade_level, polarity, subjectivity
                                 "spacy_linguistic" - Only pos,ent.dep features from spacy
                                 "top_features" - Top 15 features chosen from feature selection steps
                                 "pca" - Top 20 PCA Features
        scale_data (bool, optional): Whether or not to standard scale data. Defaults to False.
    """
    # Feature Selection
    feature_options = {}
    feature_options["submodels"] = [
        "nb_prob",
        "svm_pred",
        "ft_prob",
        "lda_t1",
        "lda_t2",
        "lda_t3",
        "lda_t4",
        "lda_t5",
    ]
    feature_options["other"] = [
        "review_stars",
        "grade_level",
        "polarity",
        "subjectivity",
    ]
    feature_options["basic_text"] = [
        "word_cnt",
        "character_cnt",
        "num_cnt",
        "uppercase_cnt",
        "#@_cnt",
        "sentence_cnt",
        "lexicon_cnt",
        "syllable_cnt",
        "avg_word_len",
        "token_cnt",
        "stopword_cnt",
        "stopword_pct",
        "ent_cnt",
        "ent_pct",
    ]
    feature_options["spacy_linguistic"] = [
        "pos_adj_pct",
        "pos_adj_cnt",
        "pos_adp_pct",
        "pos_adp_cnt",
        "pos_adv_pct",
        "pos_adv_cnt",
        "pos_aux_pct",
        "pos_aux_cnt",
        "pos_conj_pct",
        "pos_conj_cnt",
        "pos_det_pct",
        "pos_det_cnt",
        "pos_intj_pct",
        "pos_intj_cnt",
        "pos_noun_pct",
        "pos_noun_cnt",
        "pos_num_pct",
        "pos_num_cnt",
        "pos_part_pct",
        "pos_part_cnt",
        "pos_pron_pct",
        "pos_pron_cnt",
        "pos_propn_pct",
        "pos_propn_cnt",
        "pos_punct_pct",
        "pos_punct_cnt",
        "pos_sconj_pct",
        "pos_sconj_cnt",
        "pos_sym_pct",
        "pos_sym_cnt",
        "pos_verb_pct",
        "pos_verb_cnt",
        "pos_x_pct",
        "pos_x_cnt",
        "dep_root_pct",
        "dep_root_cnt",
        "dep_acl_pct",
        "dep_acl_cnt",
        "dep_acomp_pct",
        "dep_acomp_cnt",
        "dep_advcl_pct",
        "dep_advcl_cnt",
        "dep_advmod_pct",
        "dep_advmod_cnt",
        "dep_agent_pct",
        "dep_agent_cnt",
        "dep_amod_pct",
        "dep_amod_cnt",
        "dep_appos_pct",
        "dep_appos_cnt",
        "dep_attr_pct",
        "dep_attr_cnt",
        "dep_aux_pct",
        "dep_aux_cnt",
        "dep_auxpass_pct",
        "dep_auxpass_cnt",
        "dep_case_pct",
        "dep_case_cnt",
        "dep_cc_pct",
        "dep_cc_cnt",
        "dep_ccomp_pct",
        "dep_ccomp_cnt",
        "dep_compound_pct",
        "dep_compound_cnt",
        "dep_conj_pct",
        "dep_conj_cnt",
        "dep_csubj_pct",
        "dep_csubj_cnt",
        "dep_csubjpass_pct",
        "dep_csubjpass_cnt",
        "dep_dative_pct",
        "dep_dative_cnt",
        "dep_dep_pct",
        "dep_dep_cnt",
        "dep_det_pct",
        "dep_det_cnt",
        "dep_dobj_pct",
        "dep_dobj_cnt",
        "dep_expl_pct",
        "dep_expl_cnt",
        "dep_intj_pct",
        "dep_intj_cnt",
        "dep_mark_pct",
        "dep_mark_cnt",
        "dep_meta_pct",
        "dep_meta_cnt",
        "dep_neg_pct",
        "dep_neg_cnt",
        "dep_nmod_pct",
        "dep_nmod_cnt",
        "dep_npadvmod_pct",
        "dep_npadvmod_cnt",
        "dep_nsubj_pct",
        "dep_nsubj_cnt",
        "dep_nsubjpass_pct",
        "dep_nsubjpass_cnt",
        "dep_nummod_pct",
        "dep_nummod_cnt",
        "dep_oprd_pct",
        "dep_oprd_cnt",
        "dep_parataxis_pct",
        "dep_parataxis_cnt",
        "dep_pcomp_pct",
        "dep_pcomp_cnt",
        "dep_pobj_pct",
        "dep_pobj_cnt",
        "dep_poss_pct",
        "dep_poss_cnt",
        "dep_preconj_pct",
        "dep_preconj_cnt",
        "dep_predet_pct",
        "dep_predet_cnt",
        "dep_prep_pct",
        "dep_prep_cnt",
        "dep_prt_pct",
        "dep_prt_cnt",
        "dep_punct_pct",
        "dep_punct_cnt",
        "dep_quantmod_pct",
        "dep_quantmod_cnt",
        "dep_relcl_pct",
        "dep_relcl_cnt",
        "dep_xcomp_pct",
        "dep_xcomp_cnt",
        "ent_cardinal_pct",
        "ent_cardinal_cnt",
        "ent_date_pct",
        "ent_date_cnt",
        "ent_event_pct",
        "ent_event_cnt",
        "ent_fac_pct",
        "ent_fac_cnt",
        "ent_gpe_pct",
        "ent_gpe_cnt",
        "ent_language_pct",
        "ent_language_cnt",
        "ent_law_pct",
        "ent_law_cnt",
        "ent_loc_pct",
        "ent_loc_cnt",
        "ent_money_pct",
        "ent_money_cnt",
        "ent_norp_pct",
        "ent_norp_cnt",
        "ent_ordinal_pct",
        "ent_ordinal_cnt",
        "ent_org_pct",
        "ent_org_cnt",
        "ent_percent_pct",
        "ent_percent_cnt",
        "ent_person_pct",
        "ent_person_cnt",
        "ent_product_pct",
        "ent_product_cnt",
        "ent_quantity_pct",
        "ent_quantity_cnt",
        "ent_time_pct",
        "ent_time_cnt",
        "ent_work_of_art_pct",
        "ent_work_of_art_cnt",
    ]

    feature_options["top_features"] = [
        "svm_pred",
        "ft_prob",
        "nb_prob",
        "token_cnt",
        "review_stars",
        "polarity",
        "subjectivity",
        "grade_level",
        "character_cnt",
        "avg_word_len",
        "lda_t1",
        "lda_t2",
        "lda_t3",
        "lda_t4",
        "lda_t5",
    ]

    feature_options["all"] = (
        feature_options["submodels"]
        + feature_options["other"]
        + feature_options["basic_text"]
        + feature_options["spacy_linguistic"]
    )
    feature_options["non_linguistic"] = (
        feature_options["submodels"]
        + feature_options["other"]
        + feature_options["basic_text"]
    )
    feature_options["pca"] = feature_options["all"]

    features = feature_options[feature_groups] + [
        "review_id",
        "target_clf",
        "target_reg",
    ]

    train = train[features]
    test = test[features]

    # Data Split (Train/Test)
    X_train = train.drop(columns=["review_id", "target_clf", "target_reg"])
    X_test = test.drop(columns=["review_id", "target_clf", "target_reg"])
    y_train = train["target_clf"]
    y_test = test["target_clf"]

    print("\nData Split Complete")
    print(f"X_train Shape: {X_train.shape}")
    print(f"X_test Shape: {X_test.shape}")
    print(f"y_train Shape: {y_train.shape}")
    print(f"y_test Shape: {y_test.shape}")

    # Preprocessing Options
    if scale_data and feature_groups != "pca":
        start = time.perf_counter()
        standard_scaler = StandardScaler()
        X_train_scaled = standard_scaler.fit_transform(X_train)
        X_test_scaled = standard_scaler.transform(X_test)
        end = time.perf_counter()
        print("\nTrain and Test Data Scaled")
        print(f"Preprocessing took {(end-start):.2f} seconds.")
        print(f"X_train Shape: {X_train_scaled.shape}")
        print(f"X_test Shape: {X_test_scaled.shape}")
        print(f"y_train Shape: {y_train.shape}")
        print(f"y_test Shape: {y_test.shape}")
        return (X_train_scaled, X_test_scaled, y_train, y_test)
    elif feature_groups == "pca":
        start = time.perf_counter()
        standard_scaler = StandardScaler()
        X_train_scaled = standard_scaler.fit_transform(X_train)
        X_test_scaled = standard_scaler.transform(X_test)
        end = time.perf_counter()
        print("\nTrain and Test Data Scaled")
        print(f"Feature Scaling took {(end-start):.2f} seconds.")

        start = time.perf_counter()
        pca = IncrementalPCA(n_components=20)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        end = time.perf_counter()
        print("\nTrain and Test Data PCA Complete")
        print(f"PCA took {(end-start):.2f} seconds.")
        print(f"X_train Shape: {X_train_pca.shape}")
        print(f"X_test Shape: {X_test_pca.shape}")
        print(f"y_train Shape: {y_train.shape}")
        print(f"y_test Shape: {y_test.shape}")
        return (X_train_pca, X_test_pca, y_train, y_test)
    else:
        return (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    # Options
    model_naming_postfix = "_ALL_submodels_tuned"

    # Load Data
    train, test = load_data(
        ec2_or_local="local", train_rec_count=5523992, test_rec_count=1382379
    )

    # Feature Selection, Scaling, Train/Test Split
    X_train, X_test, y_train, y_test = prep_data(
        train, test, feature_groups="submodels", scale_data=True
    )

    # # Logistic Regression
    # print(">>>>>>>>>>>>>>>> Logistic Regression <<<<<<<<<<<<<<<<<<<<<<<<<")
    # log_reg_clf = LogisticRegression(random_state=7, n_jobs=-1, verbose=2)
    # log_reg_process = Base_Model_Process(log_reg_clf, "Logistic Regression")
    # log_reg_process.fit_model(X_train, y_train)
    # log_reg_process.predict_model(X_test)
    # log_reg_process.score_model(y_test)
    # # log_reg_process.save_model("log_reg", model_naming_postfix)

    # Cross Validated Logistic Regression for Hyperparameter Selection
    print(">>>>>>>>>>>>>>>> Logistic Regression CV <<<<<<<<<<<<<<<<<<<<<<<<<")
    log_reg_cv_clf = LogisticRegressionCV(random_state=7, n_jobs=-1, verbose=2)
    log_reg_cv_process = Base_Model_Process(
        log_reg_cv_clf, "Logistic Regression CV"
    )
    log_reg_cv_process.fit_model(X_train, y_train)
    log_reg_cv_process.predict_model(X_test)
    log_reg_cv_process.score_model(y_test)
    # log_reg_cv_process.save_model("log_reg_cv", model_naming_postfix)