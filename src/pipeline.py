
# * PIPELINE: CLEANING, MUNGING, ORGANIZING, FEATURE ENGINEERING
# TODO: Extended time based adjustments:
#        - Both knowable and unknowable.
# TODO: Advanced feature engineering:
#        - "Connectivity Factor" from user_friends.
#        - Creative Composite Features
# TODO: Proof for possible data leakage points.
# TODO: Convert lng/lat to x,y,z coordinates


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
from mongo_to_sql import save_to_postgres


class DataCleaningPipeline():

    def __init__(self, df):
        self.test = df.copy()

    # Set index to review_id (unique identifier for each review)
    def set_index_to_review_id(self):
        self.test.set_index('review_id', inplace=True, verify_integrity=True)

    # Delete unneeded feature columns.
    def delete_uneeded_features(self):
        columns_to_drop = ['restaurant_name', 'restaurant_address',
                           'user_name', 'restaurant_city', 'restaurant_state',
                           'restaurant_postal_code', 'user_id', 'business_id',
                           'restaurant_categories']
        self.test.drop(labels=columns_to_drop, axis=1, inplace=True)

    # Sum columns where more useful than individual columns.
    def create_feature_sums(self):
        review_positive_columns = ['review_useful', 'review_funny',
                                   'review_cool']
        user_positive_columns = ['user_useful', 'user_funny', 'user_cool']
        compliment_columns = ['user_compliment_hot', 'user_compliment_more',
                              'user_compliment_profile',
                              'user_compliment_cute',
                              'user_compliment_list', 'user_compliment_note',
                              'user_compliment_plain', 'user_compliment_cool',
                              'user_compliment_funny',
                              'user_compliment_writer',
                              'user_compliment_photos']
        self.test['review_upvotes'] = self.test[review_positive_columns].sum(axis=1)
        self.test['user_upvotes'] = self.test[user_positive_columns].sum(axis=1)
        self.test['user_compliments'] = self.test[compliment_columns].sum(axis=1)
        self.test.drop(labels=review_positive_columns, axis=1, inplace=True)
        self.test.drop(labels=user_positive_columns, axis=1, inplace=True)
        self.test.drop(labels=compliment_columns, axis=1, inplace=True)

    # Split and count list like columns.
    def count_list_features(self):
        self.test['user_friend_count'] = self.test.user_friends.apply(counter)
        self.test['user_elite_count'] = self.test.user_elite.apply(counter)
        self.test['restaurant_checkin_count'] = self.test.restaurant_checkins.apply(counter)
        self.test.drop(labels=['user_friends', 'restaurant_checkins'],
                       axis=1, inplace=True)

    # Years since last user elite.
    # TODO: Simplify and fold into other functions.
    def user_elite_feature(self):
        self.test['user_elite_most_recent'] = self.test.user_elite.apply(most_recent_elite)
        self.test['user_most_recent_elite'] = pd.to_numeric(self.test['user_elite_most_recent'])
        self.test['user_years_since_last_elite'] = 2020 - self.test['user_most_recent_elite']
        delete_elite = ['user_elite', 'user_elite_most_recent',
                        'user_most_recent_elite']
        self.test.drop(labels=delete_elite, inplace=True, axis=1)

    # Drop Nan/Null Values
    '''
    * Restaurant Price Range only column with Nan/Null values.
    * Only a small percentage were Nan/Null
    so dropping all rows with the Nan/Nulls.
    * Considered filling the Nan/Nulls
    but determined it wasn't worth the effort.
    * Other Nan/Nulls avoided during sql inner joins
    and counts of columns done above.
    '''
    def drop_nulls(self):
        self.test.dropna(inplace=True)
        self.test.drop(self.test[self.test['restaurant_price_range'] == 'None'].index, inplace=True)

    # Convert Data Types
    def convert_data_types(self):
        self.test['user_yelp_start'] = pd.to_datetime(self.test['user_yelping_since'])
        self.test['restaurant_price'] = pd.to_numeric(self.test['restaurant_price_range'])
        unused_features = ['user_yelping_since', 'restaurant_price_range']
        self.test.drop(labels=unused_features, axis=1, inplace=True)

    # Split off review text.
    def split_off_review_text(self):
        review_text = self.test['review_text']
        self.test.drop(labels=['review_text'], axis=1, inplace=True)

    # Basic feature engineering.
    def difference_between_star_counts(self):
        # review_stars - user_average_stars_given
        self.test['review_stars_v_user_avg'] = (self.test['review_stars']
                                                    - self.test['user_average_stars_given'])
        # review_stars - restaurant_overall_stars
        self.test['review_stars_v_restaurant_avg'] = (self.test['review_stars']
                                                          - self.test['restaurant_overall_stars'])
        self.test['user_days_active_at_review_time'] = (self.test['review_date']
                                                        - self.test['user_yelp_start']).dt.days

    # TARGET - Accounting for time.
    def create_target(self):
        database_creation = pd.to_datetime('2020-3-25 19:13:01')
        self.test['days_since_review'] = (database_creation
                                          - self.test.review_date).dt.days
        self.test['TARGET_review_upvotes_time_adjusted'] = (self.test['review_upvotes']
                                                            / self.test['days_since_review'])
        time_related_features_to_drop = ['days_since_review',
                                         'user_yelp_start',
                                         'review_date',
                                         'review_upvotes']
        self.test.drop(labels=time_related_features_to_drop,
                       inplace=True, axis=1)

    # Organize Data
    def organize_features(self):
        features_ordered = ['TARGET_review_upvotes_time_adjusted',
                            'review_stars', 'review_stars_v_user_avg',
                            'review_stars_v_restaurant_avg',
                            'restaurant_latitude', 'restaurant_longitude',
                            'restaurant_overall_stars',
                            'restaurant_review_count',
                            'restaurant_checkin_count', 'restaurant_is_open',
                            'restaurant_price', 'user_average_stars_given',
                            'user_review_count', 'user_upvotes',
                            'user_compliments', 'user_friend_count',
                            'user_fans', 'user_elite_count',
                            'user_years_since_last_elite',
                            'user_days_active_at_review_time']
        self.test = self.test[features_ordered]

    def full_run(self):
        self.set_index_to_review_id()
        self.delete_uneeded_features()
        self.create_feature_sums()
        self.count_list_features()
        self.user_elite_feature()
        self.drop_nulls()
        self.convert_data_types()
        self.split_off_review_text()
        self.difference_between_star_counts()
        self.create_target()
        self.organize_features()


# Helper functions for counting and finding max.
def counter(x):
    if x in ['None', None, '']:
        return 0
    else:
        y = x.split(',')
        return len(y)


def most_recent_elite(x):
    if x in ['None', None, '']:
        return 0
    else:
        y = x.split(',')
        return max(y)


if __name__ == "__main__":
    # Connection to Postgres Yelp Database - Table: restaurant_reviews_final
    connect = 'postgresql+psycopg2://postgres:password@localhost:5432/yelp'
    engine = create_engine(connect)

    chunksize = 10000
    row_count = int(pd.read_sql('SELECT COUNT(*) FROM {table_name}'.format(
        table_name='restaurant_reviews_final'), engine).values)

    total_chunks = int(row_count / chunksize) + 1
    current_chunk_num = 0
    for i in range(int(row_count / chunksize) + 1):
        query = 'SELECT * FROM {table_name} LIMIT {chunksize} OFFSET {offset}'.format(
            table_name='restaurant_reviews_final', offset=i * chunksize, chunksize=chunksize)

        chunk = pd.read_sql(query, con=engine)

        data = DataCleaningPipeline(chunk)
        data.full_run()

        data.test.to_sql('model_data_1', con=engine, index=True, if_exists='append')
        current_chunk_num += 1
        print(f'Save to Postgres Successful - Chunk Number: {current_chunk_num} of {total_chunks}')
