import pyspark as ps
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType, ArrayType

# Setting Up Spark

spark = (
    ps.sql.SparkSession.builder.appName("Yelp_ETL")
    .config(
        "spark.driver.extraClassPath", "/home/ubuntu/postgresql-42.2.20.jar"
    )
    .master("local[7]")
    .getOrCreate()
)

sc = spark.sparkContext

# Connecting To Data

data_location = "/home/ubuntu/yelp_2021/data/"
filename_prefix = "yelp_academic_dataset_"

# Checkin DataPrep

df_checkin = spark.read.json(data_location + filename_prefix + "checkin.json")
df_checkin_2 = df_checkin.withColumn(
    "date_array",
    F.split(df_checkin.date, ",").cast(ArrayType(TimestampType())),
)
df_checkin_2.createOrReplaceTempView("df_checkin_2")
df_checkin_final = spark.sql(
    """
        SELECT business_id,
            size(date_array) AS num_checkins,
            to_timestamp(array_min(date_array)) AS checkin_min,
            to_timestamp(array_max(date_array)) AS checkin_max
        FROM df_checkin_2
    """
)

# User DataPrep

df_user = spark.read.json(data_location + filename_prefix + "user.json")
df_user.createOrReplaceTempView("df_user")
df_user_1 = spark.sql(
    """
        SELECT user_id,
            to_timestamp(yelping_since) AS yelping_since,
            split(replace(elite, '20,20', '2020'), ",") AS elite_array,
            average_stars,
            review_count,
            fans,
            size(split(friends, ",")) AS friend_count,
            (compliment_cool + compliment_cute + compliment_funny
            + compliment_hot + compliment_list + compliment_more
            + compliment_note + compliment_photos + compliment_plain
            + compliment_profile + compliment_writer) AS compliments,
            (cool + funny + useful) AS ufc_count
        FROM df_user
    """
)

df_user_1.createOrReplaceTempView("df_user_1")
df_user_final = spark.sql(
    """
        SELECT user_id,
            yelping_since,
            average_stars,
            review_count,
            fans,
            friend_count,
            compliments,
            ufc_count,
            CASE
                WHEN size(elite_array) = 1
                    AND element_at(elite_array, 1) = "" THEN 0
                ELSE size(elite_array)
            END AS elite_count,
            CASE
                WHEN size(elite_array) = 1
                    AND element_at(elite_array, 1) = "" THEN 0
                ELSE int(array_min(elite_array))
            END AS elite_min,
            CASE
                WHEN size(elite_array) = 1
                    AND element_at(elite_array, 1) = "" THEN 0
                ELSE int(array_max(elite_array))
            END AS elite_max
        FROM df_user_1
    """
)

# Business DataPrep

df_business = spark.read.json(
    data_location + filename_prefix + "business.json"
)
df_business.createOrReplaceTempView("df_business")
df_business_final = spark.sql(
    """
        SELECT business_id,
            latitude,
            longitude,
            postal_code,
            state,
            stars,
            review_count
        FROM df_business
    """
)
# Review DataPrep

df_review = spark.read.json(data_location + filename_prefix + "review.json")
df_review.createOrReplaceTempView("df_review")
df_review_final = spark.sql(
    """
        SELECT review_id,
            to_timestamp(date) AS review_date,
            user_id,
            business_id,
            stars,
            text,
            ufc_total AS ufc_count_target,
        CASE
            WHEN ufc_total >= 1 THEN "True"
            ELSE "False"
        END AS ufc_bool_target
        FROM (SELECT review_id,
                    date,
                    user_id,
                    business_id,
                    stars,
                    text,
                    cool + funny + useful AS ufc_total
                FROM df_review)
    """
)

# Combining Data Tables

df_checkin_final.createOrReplaceTempView("df_checkin_final")
df_user_final.createOrReplaceTempView("df_user_final")
df_business_final.createOrReplaceTempView("df_business_final")
df_review_final.createOrReplaceTempView("df_review_final")

all_data = spark.sql(
    """
        SELECT r.review_id,
            r.user_id,
            r.business_id,
            b.latitude AS biz_latitude,
            b.longitude AS biz_longitude,
            b.postal_code AS biz_postal_code,
            b.state AS biz_state,
            b.stars AS biz_avg_stars,
            b.review_count AS biz_review_count,
            c.num_checkins AS biz_checkin_count,
            c.checkin_min AS biz_min_checkin_date,
            c.checkin_max AS biz_max_checkin_date,
            u.yelping_since AS user_yelping_since,
            u.elite_count AS user_elite_count,
            u.elite_min AS user_elite_min,
            u.elite_max AS user_elite_max,
            u.average_stars AS user_avg_stars,
            u.review_count AS user_review_count,
            u.fans AS user_fan_count,
            u.friend_count AS user_friend_count,
            u.compliments AS user_compliment_count,
            u.ufc_count AS user_ufc_count,
            r.review_date AS review_date,
            r.stars AS review_stars,
            r.text AS review_text,
            r.ufc_count_target AS target_ufc_count,
            r.ufc_bool_target AS target_ufc_bool
        FROM df_review_final AS r
        LEFT JOIN df_user_final AS u
        ON r.user_id = u.user_id
        LEFT JOIN df_business_final AS b
        ON r.business_id = b.business_id
        LEFT JOIN df_checkin_final AS c
        ON r.business_id = c.business_id
    """
)

all_data.createOrReplaceTempView("all_data")

print(f"All Data Records: {all_data.count()}")

# Split Data Into Train, Test and Holdout Sets

working_data, holdout_data = all_data.randomSplit([0.8, 0.2], seed=12345)

print(f"Working Data Records: {working_data.count()}")
print(f"Holdout Data Records: {holdout_data.count()}")

train_data, test_data = working_data.randomSplit([0.8, 0.2], seed=12345)

print(f"Train Records: {train_data.count()}")
print(f"Test Records: {test_data.count()}")

# Split Data Into Text and Non-Text

train_data.createOrReplaceTempView("train_data")
test_data.createOrReplaceTempView("test_data")

text_data_train = spark.sql(
    """
        SELECT review_id,
            review_stars,
            review_text,
            target_ufc_bool,
            target_ufc_count
        FROM train_data
    """
)

text_data_test = spark.sql(
    """
        SELECT review_id,
            review_stars,
            review_text,
            target_ufc_bool,
            target_ufc_count
        FROM test_data
    """
)

print(f"Text Train Records: {text_data_train.count()}")
print(f"Text Test Records: {text_data_test.count()}")

non_text_data_train = spark.sql(
    """
        SELECT review_id,
            user_id,
            business_id,
            review_stars,
            review_date,
            biz_avg_stars,
            biz_review_count,
            biz_checkin_count,
            biz_max_checkin_date,
            biz_min_checkin_date,
            biz_latitude,
            biz_longitude,
            biz_postal_code,
            biz_state,
            user_avg_stars,
            user_review_count,
            user_friend_count,
            user_fan_count,
            user_compliment_count,
            user_elite_count,
            user_elite_max,
            user_elite_min,
            user_yelping_since,
            target_ufc_bool,
            target_ufc_count
        FROM train_data
    """
)

non_text_data_test = spark.sql(
    """
        SELECT review_id,
            user_id,
            business_id,
            review_stars,
            review_date,
            biz_avg_stars,
            biz_review_count,
            biz_checkin_count,
            biz_max_checkin_date,
            biz_min_checkin_date,
            biz_latitude,
            biz_longitude,
            biz_postal_code,
            biz_state,
            user_avg_stars,
            user_review_count,
            user_friend_count,
            user_fan_count,
            user_compliment_count,
            user_elite_count,
            user_elite_max,
            user_elite_min,
            user_yelping_since,
            target_ufc_bool,
            target_ufc_count
        FROM test_data
    """
)

print(f"Non-Text Train Records: {non_text_data_train.count()}")
print(f"Non-Text Test Records: {non_text_data_test.count()}")

# Save Split Data To AWS RDS

db_properties = {
    "user": "postgres",
    "password": None,
    "driver": "org.postgresql.Driver",
}
db_endpoint = None
db_url = f"jdbc:postgresql://{db_endpoint}/yelp_2021_db"

text_data_train.write.jdbc(
    url=db_url,
    table="text_data_train",
    mode="overwrite",
    properties=db_properties,
)
text_data_test.write.jdbc(
    url=db_url,
    table="text_data_test",
    mode="overwrite",
    properties=db_properties,
)
non_text_data_train.write.jdbc(
    url=db_url,
    table="non_text_data_train",
    mode="overwrite",
    properties=db_properties,
)
non_text_data_test.write.jdbc(
    url=db_url,
    table="non_text_data_test",
    mode="overwrite",
    properties=db_properties,
)
holdout_data.write.jdbc(
    url=db_url,
    table="holdout_data",
    mode="overwrite",
    properties=db_properties,
)

print("ETL Complete")
