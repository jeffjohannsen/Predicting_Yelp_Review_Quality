
# ! WARNING: Memory overload issues very possible. Be careful.
# TODO: Reorganize code to speed up process and avoid memory issues.

import json
from pymongo import MongoClient
import numpy as np
import pandas as pd
from pymongo import collection
from json_to_mongo import access_specific_collection
from sqlalchemy import create_engine


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

    if batch:   # last documents
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
    df = pd.json_normalize(data, errors='ignore')
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
    print('\nBefore:\n------------------')
    print(df.info())
    new_df = df.copy()
    new_df = new_df.loc[:, columns_to_keep]
    print('\nAfter:\n------------------')
    print(new_df.info())
    return new_df


def save_to_postgres(df, yelpdb_table, action='replace', chunksize=5000):
    """
    Saves a dataframe to the yelp database on postgres.

    Args:
        df (Dataframe): Dataframe to save to postgres sql.
        yelpdb_table (string): Name of table in database to store dataframe.
        action (string): What to do if sql table already exists.
                         Options: replace (Default), append, or fail.
        chunksize (int): Number of rows to save at a time. Default 5000.
    """
    connect = 'postgresql+psycopg2://postgres:password@localhost:5432/yelp'
    engine = create_engine(connect)
    df.to_sql(yelpdb_table, con=engine, index=False,
              if_exists=action, chunksize=chunksize)
    print('Save to Postgres Successful')


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


if __name__ == "__main__":
    columns_to_keep_business = ['business_id', 'name', 'address',
                                'city', 'state', 'postal_code',
                                'latitude', 'longitude', 'stars',
                                'review_count', 'is_open', 'categories',
                                'attributes.RestaurantsPriceRange2']
    columns_to_keep_checkin = ['business_id', 'date']
    columns_to_keep_review = ['review_id', 'user_id', 'business_id', 'stars',
                              'date', 'text', 'useful', 'funny', 'cool']
    columns_to_keep_user = ['user_id', 'name', 'review_count', 'yelping_since',
                            'useful', 'funny', 'cool', 'elite', 'friends',
                            'fans', 'average_stars', 'compliment_hot',
                            'compliment_more', 'compliment_profile',
                            'compliment_cute', 'compliment_list',
                            'compliment_note', 'compliment_plain',
                            'compliment_cool', 'compliment_funny',
                            'compliment_writer', 'compliment_photos']

    # df = load_data_from_mongo('business')
    # df = trim_columns(df, columns_to_keep_business)
    # save_to_postgres(df, 'business')

    # df2 = load_data_from_mongo('checkin')
    # df2 = trim_columns(df2, columns_to_keep_checkin)
    # save_to_postgres(df2, 'checkin')

    # # Overloads memory on inbound and outbound.
    # # Switched to batch upload and downloads to fix.
    # df3 = load_data_from_mongo_in_batches('user', 5000)
    # df3 = trim_columns(df3, columns_to_keep_user)
    # save_to_postgres(df3, 'user')

    # Overloads memory even with batch setup.
    # Tried sooo many ways...
    # Had to split original json into 16 files of 500000 lines each.
    file_suffix_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08',
                        '09', '10', '11', '12', '13', '14', '15', '16']
    for file_suffix in file_suffix_list:
        filepath = f'../data/full_data/yelp_review_{file_suffix}'
        chunk = load_data_from_json(filepath)
        chunk = trim_columns(chunk, columns_to_keep_review)
        save_to_postgres(chunk, 'review', action='append', chunksize=100000)
        print(f'File {file_suffix} of 16 saved to Postgres')
