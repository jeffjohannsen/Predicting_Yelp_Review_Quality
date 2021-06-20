# Text Processing - Yelp 2021 - Part 4

# This notebook covers:
# * Word Embedding Models
#     * Word2Vec/GloVe
#     * Bert/Elmo
#     * Universal Sentence Encoder

import time

# Basic PySpark
import pyspark as ps
from pyspark.sql import functions as F
from pyspark.sql.types import *

# PySpark NLP
import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    HashingTF,
    IDF,
    StringIndexer,
    IndexToString,
    Word2Vec,
)

# Set Up Spark

spark = sparknlp.start()

# Import Data

db_properties = {
    "user": "postgres",
    "password": None,
    "driver": "org.postgresql.Driver",
}
db_endpoint = None
db_url = f"jdbc:postgresql://{db_endpoint}/yelp_2021_db"

train = spark.read.jdbc(
    url=db_url,
    table="(SELECT review_id, review_text, target_ufc_bool FROM text_data_train) AS tmp_train",
    properties=db_properties,
)
test = spark.read.jdbc(
    url=db_url,
    table="(SELECT review_id, review_text, target_ufc_bool FROM text_data_test) AS tmp_test",
    properties=db_properties,
)

print(f"Train Records: {train.count()}")
print(f"Test Records: {test.count()}")

#  Text Prep

document_assembler = (
    DocumentAssembler().setInputCol("review_text").setOutputCol("document")
)

tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("tokens")

normalizer = Normalizer().setInputCols(["tokens"]).setOutputCol("normalized")

stopwords_cleaner = (
    StopWordsCleaner()
    .setInputCols(["normalized"])
    .setOutputCol("clean_tokens")
    .setCaseSensitive(False)
)

lemmatizer = (
    LemmatizerModel.pretrained()
    .setInputCols(["clean_tokens"])
    .setOutputCol("lemma")
)

# Class Labeling

label_strIdx = StringIndexer(
    inputCol="target_ufc_bool",
    outputCol="label",
    stringOrderType="alphabetAsc",
)
label_Idxstr = IndexToString(
    inputCol="prediction",
    outputCol="predicted_class",
    labels=["False", "True"],
)

# Text Prep Options

word_embeddings = (
    WordEmbeddingsModel()
    .pretrained()
    .setInputCols(["document", "lemma"])
    .setOutputCol("word_embed")
)

# bert_embeddings = (BertEmbeddings
#                    .pretrained()
#                    .setInputCols(["document",'lemma'])
#                    .setOutputCol("word_embed"))

# elmo_embeddings = (ElmoEmbeddings
#                    .pretrained()
#                    .setInputCols(["document",'lemma'])
#                    .setOutputCol("word_embed"))

embeddings_sentence = (
    SentenceEmbeddings()
    .setInputCols(["document", "word_embed"])
    .setOutputCol("sentence_embeddings")
    .setPoolingStrategy("AVERAGE")
)

# No Preprocessing Pipelines

use = (
    UniversalSentenceEncoder.pretrained()
    .setInputCols(["document"])
    .setOutputCol("sentence_embeddings")
)

# bse = (BertSentenceEmbeddings.pretrained()
#        .setInputCols(["document"])
#        .setOutputCol("sentence_embeddings"))

# Classification Models

DL_CLF = (
    ClassifierDLApproach()
    .setInputCols("sentence_embeddings")
    .setOutputCol("prediction")
    .setLabelColumn("target_ufc_bool")
    .setMaxEpochs(25)
    .setEnableOutputLogs(True)
)

# Loading Everything to Pipeline

pipeline = Pipeline().setStages(
    [
        document_assembler,
        tokenizer,
        normalizer,
        stopwords_cleaner,
        lemmatizer,
        word_embeddings,
        embeddings_sentence,
        label_strIdx,
        DL_CLF,
    ]
)

# pipeline = (Pipeline()
#             .setStages([document_assembler,
#                         use,
#                         label_strIdx,
#                         DL_CLF,
#                        ]))


# Fit and Predict

fit_start = time.perf_counter()
cls_model = pipeline.fit(train)
fit_end = time.perf_counter()

transform_start = time.perf_counter()
test_pred = cls_model.transform(test)
test_pred = test_pred.select(
    [
        "review_id",
        "target_ufc_bool",
        "prediction.result",
        "prediction.metadata",
        "label",
    ]
)
test_pred = (
    test_pred.withColumn("prediction", test_pred["result"].getItem(0))
    .withColumn(
        "true_prob",
        test_pred["metadata"].getItem(0).getItem("True").cast("double"),
    )
    .withColumn(
        "prediction_label",
        F.udf(lambda x: 1.0 if x == "True" else 0.0, DoubleType())(
            "prediction"
        ),
    )
)
test_pred = test_pred.select(
    [
        "review_id",
        "label",
        "target_ufc_bool",
        "prediction",
        "prediction_label",
        "true_prob",
    ]
)

train_pred = cls_model.transform(test)
train_pred = train_pred.select(
    [
        "review_id",
        "target_ufc_bool",
        "prediction.result",
        "prediction.metadata",
        "label",
    ]
)
train_pred = (
    train_pred.withColumn("prediction", train_pred["result"].getItem(0))
    .withColumn(
        "true_prob",
        train_pred["metadata"].getItem(0).getItem("True").cast("double"),
    )
    .withColumn(
        "prediction_label",
        F.udf(lambda x: 1.0 if x == "True" else 0.0, DoubleType())(
            "prediction"
        ),
    )
)
train_pred = train_pred.select(
    [
        "review_id",
        "label",
        "target_ufc_bool",
        "prediction",
        "prediction_label",
        "true_prob",
    ]
)
transform_end = time.perf_counter()

print(f"Fit Time: {(fit_end - fit_start)/60:.2f} minutes")
print(f"Transform/Predict Time: {transform_end - transform_start:.2f} seconds")

# Saving Predictions

train_pred.createOrReplaceTempView("train_pred")
test_pred.createOrReplaceTempView("test_pred")

train_finished = spark.sql(
    """
                            SELECT review_id,
                                ROUND(true_prob, 3) AS glove_prob
                            FROM train_pred
                           """
)

test_finished = spark.sql(
    """
                            SELECT review_id,
                                ROUND(true_prob, 3) AS glove_prob
                            FROM test_pred
                          """
)

train_finished.write.jdbc(
    url=db_url,
    table="text_glove_train",
    mode="overwrite",
    properties=db_properties,
)
test_finished.write.jdbc(
    url=db_url,
    table="text_glove_test",
    mode="overwrite",
    properties=db_properties,
)

#  Saving Model

sc = spark.sparkContext
model_name = "GloVe_all"
cls_model.save(f"spark_models/{model_name}")
