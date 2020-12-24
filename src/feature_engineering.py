
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def data_prep_in_chunks_sql():
    """
    Runs features_and_targets table through
    Data_Finalization pipeline in chunks.
    Bypasses pandas chunksize issues
    as chunks are created in SQL.
    Provides a progress printout.
    """
    connect = 'postgresql+psycopg2://postgres:password@localhost:5432/yelp_2'
    engine = create_engine(connect)

    chunksize = 500000
    table_name = 'features_and_targets'
    row_count = int(pd.read_sql(f'SELECT COUNT(*) FROM {table_name}',
                                engine).values)

    total_chunks = int(row_count / chunksize) + 1
    current_chunk_num = 0
    for i in range(int(row_count / chunksize) + 1):
        query = f'''
                 SELECT *
                 FROM {table_name}
                 LIMIT {chunksize}
                 OFFSET {i * chunksize}
                 ;
                 '''

        chunk = pd.read_sql(query, con=engine)

        data = Data_Finalization(chunk)
        data.feature_engineering()
        data.cleaning_and_touchup()

        data.df.to_sql('non_nlp_model_data', con=engine,
                       index=False, if_exists='append')
        current_chunk_num += 1
        print(f'Chunk Number: {current_chunk_num} of {total_chunks} - '
              f'Loaded to non_nlp_model_data.')
    print('Save to Postgres Complete')


if __name__ == "__main__":
    class Data_Finalization():

        def __init__(self, df):
            self.df = df.copy()
            self.dataset_release_date = pd.to_datetime('2020-3-25 19:13:01')

        # Feature Engineering

        def difference_between_star_counts(self):
            self.test['review_stars_v_user_avg'] = (self.test['review_stars']
                                                    - self.test['user_average_stars_given'])
            self.test['review_stars_v_restaurant_avg'] = (self.test['review_stars']
                                                        - self.test['restaurant_overall_stars'])
            self.test['user_days_active_at_review_time'] = (self.test['review_date']
                                                            - self.test['user_yelp_start']).dt.days
            pass

        # Cleaning and Touchup

        def delete_unusable_features(self):
            """
            Delete unneeded feature columns
            and feature columns that could
            cause data leakage.
            """
            columns_to_drop = []
            self.df.drop(labels=columns_to_drop, axis=1, inplace=True)
            pass

        def convert_data_types(self):
            pass

        def organize_features(self):
            features_ordered = []
            self.df = self.df[features_ordered]
            pass

        def feature_engineering(self):
            # All feature engineering functions.
            pass

        def cleaning_and_touchup(self):
            # Everything else for making data model ready.
            pass
