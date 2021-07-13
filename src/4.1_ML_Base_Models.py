# Imports and Global Settings
import time
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

pd.set_option("display.float_format", lambda x: "%.5f" % x)
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)

# Options
model_naming_postfix = "_sklearn_base_model_1M"

# Load Data
start = time.perf_counter()
data_location_ec2 = "/home/ubuntu/"
data_location_local = "/home/jeff/Documents/Data_Science_Projects/Yelp_Reviews/data/full_data/model_ready/"
filepath_prefix = data_location_local

train_records_to_load = 1000000
test_records_to_load = 1000000

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
    nrows=train_records_to_load,
    true_values=["True"],
    false_values=["False"],
    dtype=datatypes,
)
test = pd.read_csv(
    f"{filepath_prefix}test.csv",
    nrows=test_records_to_load,
    true_values=["True"],
    false_values=["False"],
    dtype=datatypes,
)

end = time.perf_counter()

print("\nData Load Complete")
print(f"Took {(end-start):.2f} seconds")
print(f"Train Shape: {train.shape}")
print(f"Test Shape: {test.shape}")

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
start = time.perf_counter()
standard_scaler = StandardScaler()
X_train_scaled = standard_scaler.fit_transform(X_train)
X_test_scaled = standard_scaler.transform(X_test)
end = time.perf_counter()

print("\nTrain and Test Data Scaled")
print(f"Preprocessing took {(end-start):.2f} seconds.")


class Base_Model_Process:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.model_predictions = None
        self.model_predict_proba = None

    def fit_model(self, X_train, y_train):
        start = time.perf_counter()
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

    def score_model(self, y_test):
        start = time.perf_counter()
        cls_report = classification_report(y_test, self.model_predictions)
        auc_score = roc_auc_score(y_test, self.model_predict_proba)
        end = time.perf_counter()
        print(f"\n{self.model_name} Scoring Complete")
        print(f"Scoring took {(end-start):.2f} seconds.")
        print(f"AUC Score: {auc_score:.2f}")
        print("-----------Classification Report-----------")
        print(cls_report)

    def save_model(self, model_filename, model_name_postfix):
        dump(self.model, f"{model_filename}{model_name_postfix}.joblib")
        print(f"{self.model_name} Save Complete")
        print("===========================================")


# Logistic Regression
print(">>>>>>>>>>>>>>>> Logistic Regression <<<<<<<<<<<<<<<<<<<<<<<<<")
log_reg_clf = LogisticRegression(random_state=7, n_jobs=-1, verbose=2)
log_reg_process = Base_Model_Process(log_reg_clf, "Logistic Regression")
log_reg_process.fit_model(X_train_scaled, y_train)
log_reg_process.predict_model(X_test_scaled)
log_reg_process.score_model(y_test)
log_reg_process.save_model("log_reg", model_naming_postfix)
# Decision Tree
print(">>>>>>>>>>>>>>>> Decision Tree <<<<<<<<<<<<<<<<<<<<<<<<<")
tree_clf = DecisionTreeClassifier(random_state=7)
tree_process = Base_Model_Process(tree_clf, "Decision Tree")
tree_process.fit_model(X_train, y_train)
tree_process.predict_model(X_test)
tree_process.score_model(y_test)
tree_process.save_model("tree", model_naming_postfix)
# Random Forest
print(">>>>>>>>>>>>>>>> Random Forest <<<<<<<<<<<<<<<<<<<<<<<<<")
forest_clf = RandomForestClassifier(random_state=7, n_jobs=-1, verbose=2)
forest_process = Base_Model_Process(forest_clf, "Random Forest")
forest_process.fit_model(X_train, y_train)
forest_process.predict_model(X_test)
forest_process.score_model(y_test)
forest_process.save_model("forest", model_naming_postfix)
# XGBoost
print(">>>>>>>>>>>>>>>> XGBoost <<<<<<<<<<<<<<<<<<<<<<<<<")
xgb_clf = XGBClassifier(random_state=7, n_jobs=-1, verbosity=2)
xgb_process = Base_Model_Process(xgb_clf, "XGBoost")
xgb_process.fit_model(X_train, y_train)
xgb_process.predict_model(X_test)
xgb_process.score_model(y_test)
xgb_process.save_model("xgboost", model_naming_postfix)
