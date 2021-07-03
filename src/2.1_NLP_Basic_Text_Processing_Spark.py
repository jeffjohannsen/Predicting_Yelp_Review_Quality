# NLP - Basic Text Features | Sentiment | Reading Level

import pyspark as ps
from pyspark.sql import functions as F
from pyspark.sql.types import *

import textstat
from textblob import TextBlob

# Setting Up Spark

spark = (
    ps.sql.SparkSession.builder.appName("NLP_2.1")
    .config(
        "spark.driver.extraClassPath", "/home/ubuntu/postgresql-42.2.20.jar"
    )
    .config("spark.driver.memory", "16G")
    .master("local[7]")
    .getOrCreate()
)
sc = spark.sparkContext

# Setting up Database

db_properties = {
    "user": "postgres",
    "password": None,
    "driver": "org.postgresql.Driver",
}

db_endpoint = None
db_url = f"jdbc:postgresql://{db_endpoint}/yelp_2021_db"

# Loading Data

train = spark.read.jdbc(
    url=db_url, table="text_data_train", properties=db_properties
)
test = spark.read.jdbc(
    url=db_url, table="text_data_test", properties=db_properties
)

train.createOrReplaceTempView("train")
test.createOrReplaceTempView("test")

print(f"Train Records: {train.count()}")
print(f"Test Records: {test.count()}")

# Basic Text Characteristics


def avg_word(sentence):
    words = sentence.split()
    return sum(len(word) for word in words) / len(words)


train = (
    train.withColumn(
        "word_count",
        F.udf(lambda x: len(str(x).split(" ")), IntegerType())("review_text"),
    )
    .withColumn(
        "character_count",
        F.udf(lambda x: len(x), IntegerType())("review_text"),
    )
    .withColumn("avg_word_length", F.udf(avg_word, FloatType())("review_text"))
    .withColumn(
        "num_count",
        F.udf(
            lambda x: len([x for x in x.split() if x.isdigit()]), IntegerType()
        )("review_text"),
    )
    .withColumn(
        "uppercase_count",
        F.udf(
            lambda x: len([x for x in x.split() if x.isupper()]), IntegerType()
        )("review_text"),
    )
    .withColumn(
        "#_@_count",
        F.udf(
            lambda x: len(
                [
                    x
                    for x in x.split()
                    if x.startswith("#") or x.startswith("@")
                ]
            ),
            IntegerType(),
        )("review_text"),
    )
    .withColumn(
        "sentence_count",
        F.udf(textstat.sentence_count, IntegerType())("review_text"),
    )
    .withColumn(
        "lexicon_count",
        F.udf(textstat.lexicon_count, IntegerType())("review_text"),
    )
    .withColumn(
        "syllable_count",
        F.udf(textstat.syllable_count, IntegerType())("review_text"),
    )
)

test = (
    test.withColumn(
        "word_count",
        F.udf(lambda x: len(str(x).split(" ")), IntegerType())("review_text"),
    )
    .withColumn(
        "character_count",
        F.udf(lambda x: len(x), IntegerType())("review_text"),
    )
    .withColumn("avg_word_length", F.udf(avg_word, FloatType())("review_text"))
    .withColumn(
        "num_count",
        F.udf(
            lambda x: len([x for x in x.split() if x.isdigit()]), IntegerType()
        )("review_text"),
    )
    .withColumn(
        "uppercase_count",
        F.udf(
            lambda x: len([x for x in x.split() if x.isupper()]), IntegerType()
        )("review_text"),
    )
    .withColumn(
        "#_@_count",
        F.udf(
            lambda x: len(
                [
                    x
                    for x in x.split()
                    if x.startswith("#") or x.startswith("@")
                ]
            ),
            IntegerType(),
        )("review_text"),
    )
    .withColumn(
        "sentence_count",
        F.udf(textstat.sentence_count, IntegerType())("review_text"),
    )
    .withColumn(
        "lexicon_count",
        F.udf(textstat.lexicon_count, IntegerType())("review_text"),
    )
    .withColumn(
        "syllable_count",
        F.udf(textstat.syllable_count, IntegerType())("review_text"),
    )
)

# Reading Level

train = train.withColumn(
    "grade_level",
    F.udf(textstat.flesch_kincaid_grade, FloatType())("review_text"),
)
test = test.withColumn(
    "grade_level",
    F.udf(textstat.flesch_kincaid_grade, FloatType())("review_text"),
)

# Sentiment Analysis

train = train.withColumn(
    "polarity",
    F.udf(lambda x: TextBlob(x).sentiment.polarity, FloatType())(
        "review_text"
    ),
).withColumn(
    "subjectivity",
    F.udf(lambda x: TextBlob(x).sentiment.subjectivity, FloatType())(
        "review_text"
    ),
)
test = test.withColumn(
    "polarity",
    F.udf(lambda x: TextBlob(x).sentiment.polarity, FloatType())(
        "review_text"
    ),
).withColumn(
    "subjectivity",
    F.udf(lambda x: TextBlob(x).sentiment.subjectivity, FloatType())(
        "review_text"
    ),
)

# Save Data To AWS RDS

train.write.jdbc(
    url=db_url,
    table="text_data_train_b",
    mode="overwrite",
    properties=db_properties,
)
test.write.jdbc(
    url=db_url,
    table="text_data_test_b",
    mode="overwrite",
    properties=db_properties,
)
