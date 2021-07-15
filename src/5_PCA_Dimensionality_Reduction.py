# Principal Component Analysis Dimensionality Reduction

# Imports and Global Settings
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Loading Data
start = time.perf_counter()

# EC2
filepath_prefix = "/home/ubuntu/"
# Local
# filepath_prefix = "/home/jeff/Documents/Data_Science_Projects/Yelp_Reviews/data/full_data/model_ready/"

train_records_to_load = 10000
test_records_to_load = 1000

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

# Scale Data
start = time.perf_counter()

standard_scaler = StandardScaler()
X_train_scaled = standard_scaler.fit_transform(X_train)
X_test_scaled = standard_scaler.transform(X_test)

end = time.perf_counter()

print("\nTrain and Test Data Scaled")
print(f"Preprocessing took {(end-start):.2f} seconds.")

# Perform PCA
start = time.perf_counter()

num_components = 5

pca = PCA(n_components=num_components)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

end = time.perf_counter()

print(f"PCA Fit and Transform Complete")
print(f"PCA took {(end-start)/60:.2f} minutes.")

# Explained Variance
explained_variance = pca.explained_variance_ratio_
print(f"{explained_variance.sum()*100:.1f}% of the variance is explained.")

# Create New Train and Test Dataframes
train_id_target = train[["review_id", "target_clf", "target_reg"]]
train_principal_components = pd.DataFrame(
    X_train_pca, columns=[f"pc{i}" for i in range(1, num_components + 1)]
)
train_pca = pd.concat([train_id_target, train_principal_components], axis=1)

test_id_target = test[["review_id", "target_clf", "target_reg"]]
test_principal_components = pd.DataFrame(
    X_test_pca, columns=[f"pc{i}" for i in range(1, num_components + 1)]
)
test_pca = pd.concat([test_id_target, test_principal_components], axis=1)

# Save to CSV
train_pca.to_csv(f"{filepath_prefix}train_pca.csv", index=False)
test_pca.to_csv(f"{filepath_prefix}test_pca.csv", index=False)
