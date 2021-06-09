# Text Processing - Yelp 2021 - Part 2
# This file covers:
# Linguistic Characterics (parts-of-speech, named entities,
#                          syntactic relationships - Spacy)
# Imports and Global Settings
# Common Libraries
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Main NLP library
import spacy

# Connecting to Postgres RDS on AWS
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql

pd.set_option("display.float_format", lambda x: "%.5f" % x)

# Import Data in Chunks

db_endpoint = None
db_password = None

engine = create_engine(
    f"postgresql+psycopg2://postgres:{db_password}@{db_endpoint}/yelp_2021_db"
)
conn = engine.connect().execution_options(stream_results=True)
chunksize = 100000

# Linguistic Components with Spacy

pos_list = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]
dep_list = [
    "ROOT",
    "acl",
    "acomp",
    "advcl",
    "advmod",
    "agent",
    "amod",
    "appos",
    "attr",
    "aux",
    "auxpass",
    "case",
    "cc",
    "ccomp",
    "compound",
    "conj",
    "csubj",
    "csubjpass",
    "dative",
    "dep",
    "det",
    "dobj",
    "expl",
    "intj",
    "mark",
    "meta",
    "neg",
    "nmod",
    "npadvmod",
    "nsubj",
    "nsubjpass",
    "nummod",
    "oprd",
    "parataxis",
    "pcomp",
    "pobj",
    "poss",
    "preconj",
    "predet",
    "prep",
    "prt",
    "punct",
    "quantmod",
    "relcl",
    "xcomp",
]
ent_list = [
    "CARDINAL",
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PERSON",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "WORK_OF_ART",
]

nlp = spacy.load("en_core_web_sm")


def create_spacy_features(df, text_feature_name):
    """
        Adds various features using Spacy's library and NLP models.
    ​
        Key Terms:
            pos_dict: Part of Speech
                      https://universaldependencies.org/u/pos/
    ​
            dep_list: Universal Dependency Relations
                      https://universaldependencies.org/u/dep/
    ​
            ent_list: Named Entity
                      https://spacy.io/api/annotation#named-entities
    """

    df["spacy_doc"] = df[text_feature_name].apply(lambda x: nlp(x))
    df.drop("review_text", axis=1, inplace=True)

    df["token_count"] = df["spacy_doc"].apply(lambda x: len(x))
    df["stopword_perc"] = df["spacy_doc"].apply(
        lambda x: round(
            len([token for token in x if token.is_stop]) / len(x), 5
        )
    )
    df["stopword_count"] = df["spacy_doc"].apply(
        lambda x: len([token for token in x if token.is_stop])
    )
    df["ent_perc"] = df["spacy_doc"].apply(
        lambda x: round(len(x.ents) / len(x), 5)
    )
    df["ent_count"] = df["spacy_doc"].apply(lambda x: len(x.ents))

    for pos in pos_list:
        df[f"pos_{pos.lower()}_perc"] = df["spacy_doc"].apply(
            lambda x: round(
                len([token for token in x if token.pos_ == pos]) / len(x), 5
            )
        )
        df[f"pos_{pos.lower()}_count"] = df["spacy_doc"].apply(
            lambda x: len([token for token in x if token.pos_ == pos])
        )
    for dep in dep_list:
        df[f"dep_{dep.lower()}_perc"] = df["spacy_doc"].apply(
            lambda x: round(
                len([token for token in x if token.dep_ == dep]) / len(x), 5
            )
        )
        df[f"dep_{dep.lower()}_count"] = df["spacy_doc"].apply(
            lambda x: len([token for token in x if token.dep_ == dep])
        )
    for ent in ent_list:
        df[f"ent_{ent.lower()}_perc"] = df["spacy_doc"].apply(
            lambda x: round(
                len([y for y in x.ents if y.label_ == ent]) / len(x), 5
            )
        )
        df[f"ent_{ent.lower()}_count"] = df["spacy_doc"].apply(
            lambda x: len([y for y in x.ents if y.label_ == ent])
        )

    df.drop("spacy_doc", axis=1, inplace=True)

    return df


# Run Spacy Function and Save to AWS RDS

records_processed = 0
for chunk in pd.read_sql(
    sql="SELECT review_id, review_text FROM text_data_train",
    con=conn,
    chunksize=chunksize,
):
    start = time.perf_counter()
    text = create_spacy_features(chunk, "review_text")
    records_processed += text.shape[0]
    text.to_sql(
        "text_data_train_spacy", con=engine, index=False, if_exists="append"
    )
    stop = time.perf_counter()
    print(f"Total records processed: {records_processed}")
    print(f"Loop time: {((stop-start) / 60):.2f} minutes")


records_processed = 0
for chunk in pd.read_sql(
    sql="SELECT review_id, review_text FROM text_data_test",
    con=conn,
    chunksize=chunksize,
):
    start = time.perf_counter()
    text = create_spacy_features(chunk, "review_text")
    records_processed += text.shape[0]
    text.to_sql(
        "text_data_test_spacy", con=engine, index=False, if_exists="append"
    )
    stop = time.perf_counter()
    print(f"Total records processed: {records_processed}")
    print(f"Loop time: {((stop-start) / 60):.2f} minutes")
