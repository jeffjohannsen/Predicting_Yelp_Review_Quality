# Imports
import csv
import fasttext
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from gensim.utils import simple_preprocess

# Load Data
db_endpoint = None
db_name = "yelp_2021_db"
db_password = None

engine = create_engine(
    f"postgresql+psycopg2://postgres:{db_password}@{db_endpoint}/{db_name}"
)

train_query = (
    "SELECT review_id, review_text, target_ufc_bool FROM text_data_train"
)
test_query = (
    "SELECT review_id, review_text, target_ufc_bool FROM text_data_test"
)

train = pd.read_sql(sql=train_query, con=engine)
test = pd.read_sql(sql=test_query, con=engine)

print("Data Loaded")

# Preprocessing
train["review_text"] = train["review_text"].apply(
    lambda x: " ".join(simple_preprocess(x))
)
train["target_ufc_bool"] = train["target_ufc_bool"].apply(
    lambda x: "__label__" + x
)

test["review_text"] = test["review_text"].apply(
    lambda x: " ".join(simple_preprocess(x))
)
test["target_ufc_bool"] = test["target_ufc_bool"].apply(
    lambda x: "__label__" + x
)

print("Preprocess Complete")

train[["target_ufc_bool", "review_text"]].to_csv(
    "ft_train.txt",
    index=False,
    sep=" ",
    header=None,
    quoting=csv.QUOTE_NONE,
    quotechar="",
    escapechar=" ",
)

test[["target_ufc_bool", "review_text"]].to_csv(
    "ft_test.txt",
    index=False,
    sep=" ",
    header=None,
    quoting=csv.QUOTE_NONE,
    quotechar="",
    escapechar=" ",
)

print("Temp Files Saved")

# Train Model
ft_model = fasttext.train_supervised(input="ft_train.txt", loss="hs")

# Test Model
print(ft_model.test("ft_test.txt"))

print("Train and Test Complete")

# Get Predictions
train["predictions"] = train["review_text"].apply(ft_model.predict)
test["predictions"] = test["review_text"].apply(ft_model.predict)


def get_quality_prob(x):
    if x[0][0] == "__label__True":
        return round(x[1][0], 5)
    elif x[0][0] == "__label__False":
        return round(1 - x[1][0], 5)


train["ft_quality_prob"] = train["predictions"].apply(get_quality_prob)
test["ft_quality_prob"] = test["predictions"].apply(get_quality_prob)

train = train.drop(columns=["review_text", "target_ufc_bool", "predictions"])
test = test.drop(columns=["review_text", "target_ufc_bool", "predictions"])

print("Predictions Complete")

# Save Fasttext Results and Model
train.to_sql(
    "text_fasttext_train", con=engine, index=False, if_exists="replace"
)
test.to_sql("text_fasttext_test", con=engine, index=False, if_exists="replace")

print("Save to RDS Complete")

ft_model.save_model("fasttext_model_ALL")

print("Model Save Complete")
print("Done")
print("-------------------------------------")
