# Text Processing - Yelp 2021 - Part 3
# * Tf-Idf Text Vectorization
# * Naive Bayes Predictions
# * Support Vector Machine Predictions

# Imports and Global Settings
import time

# Basic PySpark
import pyspark as ps
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.functions import vector_to_array

# PySpark NLP
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, StringIndexer, IndexToString

# PySpark Classification Models
from pyspark.ml.classification import NaiveBayes, LinearSVC

# PySpark Model Evaluation
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

# Stopwords
from nltk.corpus import stopwords

# Set Up Spark

spark = (
    ps.sql.SparkSession.builder.appName("Spark NLP")
    .master("local[3]")
    .config("spark.driver.memory", "16G")
    .config("spark.driver.maxResultSize", "0")
    .config("spark.kryoserializer.buffer.max", "2000M")
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.1.0")
    .config(
        "spark.driver.extraClassPath", "/home/jovyan/postgresql-42.2.20.jar"
    )
    .getOrCreate()
)

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
    table="(SELECT review_id, review_text, target_ufc_bool FROM text_data_train LIMIT 1000) AS tmp_train",
    properties=db_properties,
)
test = spark.read.jdbc(
    url=db_url,
    table="(SELECT review_id, review_text, target_ufc_bool FROM text_data_test LIMIT 1000) AS tmp_test",
    properties=db_properties,
)

train.createOrReplaceTempView("train")
test.createOrReplaceTempView("test")

# Data Overview

print(f"Train Records: {train.count()}")
print(f"Test Records: {test.count()}")

# Text Processing Pipeline Setup

eng_stopwords = stopwords.words("english")

document_assembler = (
    DocumentAssembler().setInputCol("review_text").setOutputCol("document")
)
sentence = (
    SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
)
tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")
normalizer = (
    Normalizer()
    .setInputCols(["token"])
    .setOutputCol("normalized")
    .setLowercase(True)
)
lemmatizer = (
    LemmatizerModel.pretrained()
    .setInputCols(["normalized"])
    .setOutputCol("lemma")
)
stopwords_cleaner = (
    StopWordsCleaner()
    .setInputCols(["lemma"])
    .setOutputCol("clean_lemma")
    .setCaseSensitive(False)
    .setStopWords(eng_stopwords)
)
finisher = (
    Finisher()
    .setInputCols(["clean_lemma"])
    .setOutputCols(["token_features"])
    .setOutputAsArray(True)
    .setCleanAnnotations(False)
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

# Text Vectorization

hashTF = HashingTF(inputCol="token_features", outputCol="tf_features")
idf = IDF(inputCol="tf_features", outputCol="features", minDocFreq=2)

# Classification Models

mnb_clf = NaiveBayes(smoothing=1.0)
svm_clf = LinearSVC(standardization=False)

#  Loading Everything to Pipeline

pipeline = Pipeline().setStages(
    [
        document_assembler,
        sentence,
        tokenizer,
        normalizer,
        lemmatizer,
        stopwords_cleaner,
        finisher,
        hashTF,
        idf,
        label_strIdx,
        svm_clf,
        label_Idxstr,
    ]
)

# Fit and Predict

fit_start = time.perf_counter()
cls_model = pipeline.fit(train)
fit_end = time.perf_counter()

transform_start = time.perf_counter()
test_pred = cls_model.transform(test)
train_pred = cls_model.transform(train)
transform_end = time.perf_counter()

# Saving Predictions

# Naive Bayes

# train_pred = train_pred.withColumn("Prob", vector_to_array("probability"))
# test_pred = test_pred.withColumn("Prob", vector_to_array("probability"))

# train_pred.createOrReplaceTempView("train_pred")
# test_pred.createOrReplaceTempView("test_pred")

# train_finished = spark.sql(
#     """
#       SELECT review_id,
#           ROUND(Prob[1], 3) AS NB_tfidf_true_prob
#       FROM train_pred
#     """
# )

# test_finished = spark.sql(
#     """
#         SELECT review_id,
#             ROUND(Prob[1], 3) AS NB_tfidf_true_prob
#         FROM test_pred
#     """
# )

# train_finished.write.jdbc(
#     url=db_url,
#     table="text_data_train_nm_tfidf",
#     mode="overwrite",
#     properties=db_properties,
# )
# test_finished.write.jdbc(
#     url=db_url,
#     table="text_data_test_nm_tfidf",
#     mode="overwrite",
#     properties=db_properties,
# )

# SVM

train_pred = train_pred.withColumn(
    "rawPrediction", vector_to_array("rawPrediction")
)
test_pred = test_pred.withColumn(
    "rawPrediction", vector_to_array("rawPrediction")
)

train_pred.createOrReplaceTempView("train_pred")
test_pred.createOrReplaceTempView("test_pred")

train_finished = spark.sql(
    """
        SELECT review_id,
            ROUND(rawPrediction[1], 3) AS svm_pred
        FROM train_pred
    """
)

test_finished = spark.sql(
    """
        SELECT review_id,
            ROUND(rawPrediction[1], 3) AS svm_pred
        FROM train_pred
    """
)

train_finished.write.jdbc(
    url=db_url,
    table="text_data_train_svm_tfidf",
    mode="overwrite",
    properties=db_properties,
)
test_finished.write.jdbc(
    url=db_url,
    table="text_data_test_svm_tfidf",
    mode="overwrite",
    properties=db_properties,
)

# Save Model

sc = spark.sparkContext
model_name = "SVM_TFIDF_1k"
cls_model.save(f"spark_models/{model_name}")
