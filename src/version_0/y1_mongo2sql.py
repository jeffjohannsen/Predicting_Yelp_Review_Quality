# ! WARNING: Memory overload issues very possible. Be careful.
# TODO: Simplify and condense functions.
# Deprecated code. Use new json2sql.py setup.
# Current version bypasses MongoDB.
# This is starter code for returning to MongoDB if necessary.


import numpy as np
import pandas as pd
from pymongo import MongoClient, collection
from sqlalchemy import create_engine

from y1_json2mongo import access_specific_collection


def batched(cursor, batch_size):
    """
    Splits MongoDB query into batches to be loaded into Pandas.
    Avoids memory usage errors.

    Args:
        cursor (Database Connection): Search results of a mongodb query.
        batch_size (int): Number to records to load in at one time.
                          Defaults to 5000. Depends on memory usage.

    Yields:
        list: List of batch_size number of records.
    """
    batch = []
    for doc in cursor:
        batch.append(doc)
        if batch and not len(batch) % batch_size:
            yield batch
            batch = []

    if batch:  # last documents
        yield batch


def load_data_from_mongo(collection_name):
    """
    Load data from specified collection
    within mongo yelp database
    into pandas dataframe.

    Args:
        collection_name (string): Name of collection to load.

    Returns:
        Dataframe: Semi-flattened mongo collection.
    """
    collection = access_specific_collection(collection_name)
    data = list(collection.find({}))
    df = pd.json_normalize(data, errors="ignore")
    print(df.head(5))
    return df


def load_data_from_mongo_in_batches(collection_name, batch_size=5000):
    """
    Load data from specified collection
    within mongo yelp database
    into pandas dataframe.
    One batch at a time to avoid memory use errors.

    Args:
        collection_name (string): Name of collection to load.
        batch_size (int): Number to records to load in at one time.
                          Defaults to 5000. Depends on memory usage.

    Returns:
        Dataframe: Semi-flattened mongo collection.
    """
    collection = access_specific_collection(collection_name)
    cursor = collection.find()
    df = pd.DataFrame()
    for batch in batched(cursor, 5000):
        df = df.append(batch, ignore_index=True)
    return df


def trim_columns(df, columns_to_keep):
    """
    Choose which columns to keep for a first pass
    at data size reduction.

    Args:
        df (Dataframe): Dataframe from which to choose columns.
        columns_to_keep (list of strings): Columns from dataframe to keep.

    Returns:
        Dataframe: Original dataframe minus columns not kept.
    """
    print("\nBefore:\n------------------")
    print(df.info())
    new_df = df.copy()
    new_df = new_df.loc[:, columns_to_keep]
    print("\nAfter:\n------------------")
    print(new_df.info())
    return new_df


def save_to_postgres(df, db, table, action="replace", chunksize=5000):
    """
    Saves a dataframe to a database on postgres.

    Args:
        df (Dataframe): Dataframe to save to postgres sql.
        db (string): Name of database to store dataframe.
        table (string): Name of table in database to store dataframe.
        action (string): What to do if sql table already exists.
                         Options: replace (Default), append, or fail.
        chunksize (int): Number of rows to save at a time. Default 5000.
    """
    connect = f"postgresql+psycopg2://postgres:password@localhost:5432/{db}"
    engine = create_engine(connect)
    df.to_sql(
        table, con=engine, index=False, if_exists=action, chunksize=chunksize
    )
    print("Save to Postgres Successful")


def load_data_from_json(filepath):
    """
    Wrapper for pandas read_json.
    Used for updating parameters and process
    if necessary in the future.

    Args:
        filepath (string): Filepath to json file to be loaded.

    Returns:
        Dataframe: Pandas df containing data.
    """
    data = pd.read_json(filepath, lines=True)
    return data


def convert_business_data(columns):
    """
    Converts business collection on Mongo
    to business table on Postgres.

    Args:
        columns (list): Columns of collection to transfer
                        to table.
    """
    df = load_data_from_mongo("business")
    df = trim_columns(df, columns)
    save_to_postgres(df, "business")


def convert_checkin_data(columns):
    """
    Converts checkin collection on Mongo
    to checkin table on Postgres.

    Args:
        columns (list): Columns of collection to transfer
                        to table.
    """
    df = load_data_from_mongo("checkin")
    df = trim_columns(df, columns)
    save_to_postgres(df, "checkin")


# # Overloads memory on inbound and outbound.
# # Switched to batch upload and downloads to fix.
def convert_user_data(columns):
    """
    Converts user collection on Mongo
    to user table on Postgres.

    Args:
        columns (list): Columns of collection to transfer
                        to table.
    """
    df = load_data_from_mongo_in_batches("user", 5000)
    df = trim_columns(df, columns)
    save_to_postgres(df, "user")


# Overloads memory even with batch setup.
# Have to split json into smaller json files.
def convert_review_data(files, columns):
    """
    Converts review collection on Mongo
    to review table on Postgres.

    Args:
        files (list): Filenames of split json files to be converted.
        columns (list): Columns of collection to transfer
                        to table.
    """
    file_num = 0
    for file in files:
        filepath = f"../data/full_data/{file}"
        chunk = load_data_from_json(filepath)
        chunk = trim_columns(chunk, columns)
        save_to_postgres(chunk, "review", action="append", chunksize=100000)
        file_num += 1
        print(f"{file} saved to Postgres. {file_num} of {len(files)}")


if __name__ == "__main__":
    columns_to_keep_business = [
        "business_id",
        "name",
        "address",
        "city",
        "state",
        "postal_code",
        "latitude",
        "longitude",
        "stars",
        "review_count",
        "is_open",
        "categories",
        "attributes.RestaurantsPriceRange2",
    ]
    columns_to_keep_checkin = ["business_id", "date"]
    columns_to_keep_review = [
        "review_id",
        "user_id",
        "business_id",
        "stars",
        "date",
        "text",
        "useful",
        "funny",
        "cool",
    ]
    columns_to_keep_user = [
        "user_id",
        "name",
        "review_count",
        "yelping_since",
        "useful",
        "funny",
        "cool",
        "elite",
        "friends",
        "fans",
        "average_stars",
        "compliment_hot",
        "compliment_more",
        "compliment_profile",
        "compliment_cute",
        "compliment_list",
        "compliment_note",
        "compliment_plain",
        "compliment_cool",
        "compliment_funny",
        "compliment_writer",
        "compliment_photos",
    ]

    # Split review json file in command line.
    review_file_list = []

    convert_business_data(columns_to_keep_business)
    convert_checkin_data(columns_to_keep_checkin)
    convert_user_data(columns_to_keep_user)
    convert_review_data(review_file_list, columns_to_keep_review)
