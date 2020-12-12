
import json
import pandas as pd
from sqlalchemy import create_engine


# TODO: Alter function once Pandas fixes their bug.
def yelp_json_to_sql(file, flatten=True):
    """
    Loads json into postgres database.

    Args:
        file (string): Identifying suffix of yelp json file.
        flatten (bool, optional): If true flattens nested json.
                                  Defaults to True.
    """
    filepath = f'../data/full_data/yelp_academic_dataset_{file}.json'
    connect = 'postgresql+psycopg2://postgres:password@localhost:5432/yelp_2'
    engine = create_engine(connect)

    chunk_size = 50000
    batch_num = 0
    # Pandas open bug with nrows makes chunking not work.
    # Set nrows to number > total number of rows in file to overcome bug.
    # Bug fix in progress as of 12/9/2020.
    # https://github.com/pandas-dev/pandas/pull/38293
    for chunk in pd.read_json(filepath, chunksize=chunk_size,
                              orient="records", lines=True, nrows=100000000):
        data = chunk.copy()
        if flatten:
            data = json.loads(data.to_json(orient="records"))
            data = pd.json_normalize(data, sep='_', max_level=None)
        data.to_sql(f'{file}', engine, if_exists='append', index=False)
        batch_num += 1
        print(f'Batch Number: {batch_num} - Loaded to table {file}.')


if __name__ == "__main__":
    file_suffix_list = ['business', 'user', 'review', 'checkin', 'tip']

    for file in file_suffix_list:
        yelp_json_to_sql(file)
