
import pandas as pd
from sqlalchemy import create_engine


def yelp_json_to_sql(file, flatten=True):
    """
    Loads json into postgres database.

    Args:
        file (string): Identifying suffix of yelp json file.
        flatten (bool, optional): If true flattens nested json into one level.
                                  Defaults to True.
    """
    filepath = f'../data/full_data/yelp_academic_dataset_{file}.json'
    connect = 'postgresql+psycopg2://postgres:password@localhost:5432/yelp_2'
    engine = create_engine(connect)

    chunk_size = 50000
    batch_num = 0
    # pandas bug with nrows makes chunking not work.
    # set nrows to number > total number of rows in file to overcome bug.
    for chunk in pd.read_json(filepath, chunksize=chunk_size,
                              orient="records", lines=True,
                              nrows=100000000):
        if flatten:
            chunk = pd.json_normalize(chunk, sep='_')
        chunk.to_sql(f'json_{file}', engine, if_exists='append')
        batch_num += 1
        print(f'Batch Number: {batch_num} Loaded to table json_{file}.')


if __name__ == "__main__":
    file_suffix_list = ['business', 'user', 'review', 'checkin', 'tip']
    for file in file_suffix_list:
        yelp_json_to_sql(file, flatten=False)
