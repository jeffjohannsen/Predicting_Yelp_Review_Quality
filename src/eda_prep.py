
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


class EDA_Prep():

    def __init__(self, df):
        self.df = df.copy()
        self.dataset_release_date = pd.to_datetime('2020-3-25 19:13:01')

    # Helper Functions

    def usefulness_level(self, x):
        if x == 0:
            return 'zero'
        elif x < 10:
            return 'low'
        elif x < 100:
            return 'medium'
        elif x >= 100:
            return 'high'
        else:
            return 'unknown'

    def target_time_discount(self, ufc_total, review_date):
        return (ufc_total / ((self.dataset_release_date
                              - review_date).days)) * 365

    def count_elite(self, x):
        elite_count = 0
        if x in ['None', None, '']:
            elite_count = 0
        else:
            y = x.split(',')
            elite_count = len(y)
        return elite_count

    def years_since_most_recent_elite(self, x):
        z = 0
        if x in ['None', None, '']:
            z = 0
        else:
            y = pd.to_numeric(x.split(','))
            z = max(y)
        return 2020 - z

    def count_elite_td(self, user_elite, review_date):
        if user_elite in ['None', None, '']:
            return 0
        else:
            split_elites = user_elite.split(',')
            elites_pre_review = [elite for elite in split_elites
                                 if pd.to_datetime(elite) < review_date]
            return len(elites_pre_review)

    def years_since_most_recent_elite_td(self, user_elite, review_date):
        z = 0
        if user_elite in ['None', None, '']:
            z = 0
        else:
            split_elites = pd.to_numeric(user_elite.split(','))
            elites_pre_review = [elite for elite in split_elites
                                 if pd.to_datetime(elite) < review_date]
            if len(elites_pre_review) == 0:
                z = 0
            else:
                z = max(elites_pre_review)
        return review_date.year - z

    def user_time_discount(self, count_feature,
                           user_yelping_since, review_date):
        return (count_feature / (self.dataset_release_date
                                 - user_yelping_since).days) \
                * ((review_date - user_yelping_since).days)

    def business_time_discount(self, count_feature,
                               oldest_checkin, review_date):
        return (count_feature / (self.dataset_release_date
                                 - oldest_checkin).days) \
                * ((review_date - oldest_checkin).days)

    # Central Functionality

    def convert_to_datetime(self):
        self.df['review_date'] = pd.to_datetime(self.df['review_date'],
                                                unit='ms')
        self.df['user_yelping_since'] = \
            pd.to_datetime(self.df['user_yelping_since'])

    def create_targets(self):
        # No discounting or scaling
        self.df['T1_REG_review_total_ufc'] = \
            self.df[['review_useful',
                     'review_funny',
                     'review_cool']].sum(axis=1)
        self.df['T2_CLS_ufc_>0'] = self.df['T1_REG_review_total_ufc'] > 0
        self.df['T3_CLS_ufc_level'] = \
            self.df['T1_REG_review_total_ufc'].apply(self.usefulness_level)
        # Time discounted
        self.df['T4_REG_ufc_TD'] = \
            self.df.apply(lambda x:
                          self.target_time_discount(x.T1_REG_review_total_ufc,
                                                    x.review_date), axis=1)
        self.df['T5_CLS_ufc_level_TD'] = \
            self.df['T4_REG_ufc_TD'].apply(self.usefulness_level)
        # Time and Business Popularity Discounted
        self.df['T6_REG_ufc_TDBD'] = \
            self.df['T4_REG_ufc_TD'] / self.df['business_review_count']
        self.df.drop(labels=['review_useful', 'review_funny', 'review_cool'],
                     axis=1, inplace=True)

    def remove_nan_null(self):
        self.df.dropna(inplace=True)

    def combine_features(self):
        compliment_columns = \
            ['user_compliment_hot', 'user_compliment_more',
             'user_compliment_profile', 'user_compliment_cute',
             'user_compliment_list', 'user_compliment_note',
             'user_compliment_plain', 'user_compliment_cool',
             'user_compliment_funny', 'user_compliment_writer',
             'user_compliment_photos']
        self.df['user_compliments'] = self.df[compliment_columns].sum(axis=1)
        self.df.drop(labels=compliment_columns, axis=1, inplace=True)
        self.df['user_total_ufc'] = self.df[['user_useful',
                                             'user_funny',
                                             'user_cool']].sum(axis=1)
        self.df.drop(labels=['user_useful', 'user_funny', 'user_cool'],
                     axis=1, inplace=True)

    def create_user_elite_basic_features(self):
        self.df['user_elite_count'] = \
            self.df.user_elite.apply(self.count_elite)
        self.df['user_most_recent_elite'] = \
            self.df.user_elite.apply(self.years_since_most_recent_elite)

    def create_user_elite_time_discounted(self):
        self.df['user_elite_count_TD'] = \
            self.df.apply(lambda x: self.count_elite_td(x.user_elite,
                                                        x.review_date), axis=1)
        self.df['user_most_recent_elite_TD'] = \
            self.df.apply(lambda x:
                          self.years_since_most_recent_elite_td(x.user_elite,
                                                                x.review_date),
                          axis=1)

    def time_discount_user_features(self):
        user_features_needing_time_discounting = ['user_total_ufc',
                                                  'user_compliments',
                                                  'user_review_count',
                                                  'user_fans',
                                                  'user_friend_count']
        for feature in user_features_needing_time_discounting:
            self.df[f'{feature}_TD'] = \
                self.df.apply(lambda x:
                              self.user_time_discount(x[feature],
                                                      x.user_yelping_since,
                                                      x.review_date), axis=1)

    def time_discount_business_features(self):
        business_features_needing_time_discounting = ['business_review_count',
                                                      'business_checkin_count']
        for feature in business_features_needing_time_discounting:
            self.df[f'{feature}_TD'] = \
                self.df.apply(lambda x:
                              self.business_time_discount(x[feature],
                                                          x.business_oldest_checkin,
                                                          x.review_date),
                              axis=1)

    def organize(self):
        features_ordered = ['review_id', 'user_id', 'business_id',
                            'review_stars', 'review_date',
                            'business_avg_stars', 'business_review_count',
                            'business_checkin_count', 'business_is_open',
                            'business_categories', 'business_oldest_checkin',
                            'business_newest_checkin', 'business_latitude',
                            'business_longitude', 'business_review_count_TD',
                            'business_checkin_count_TD', 'user_avg_stars',
                            'user_total_ufc', 'user_review_count',
                            'user_friend_count', 'user_fans',
                            'user_compliments', 'user_elite_count',
                            'user_most_recent_elite', 'user_yelping_since',
                            'user_total_ufc_TD', 'user_review_count_TD',
                            'user_friend_count_TD', 'user_fans_TD',
                            'user_compliments_TD', 'user_elite_count_TD',
                            'user_most_recent_elite_TD',
                            'T1_REG_review_total_ufc', 'T2_CLS_ufc_>0',
                            'T3_CLS_ufc_level', 'T4_REG_ufc_TD',
                            'T5_CLS_ufc_level_TD', 'T6_REG_ufc_TDBD']
        self.df = self.df[features_ordered]

    def run_all(self):
        self.convert_to_datetime()
        self.create_targets()
        self.remove_nan_null()
        self.combine_features()
        self.create_user_elite_basic_features()
        self.create_user_elite_time_discounted()
        self.time_discount_user_features()
        self.time_discount_business_features()
        self.organize()


def data_prep_in_chunks_sql():
    """
    Runs all_features table through
    EDA_Prep pipeline in chunks.
    Bypasses pandas chunksize issues
    as chunks are created in SQL.
    Provides a progress printout.
    """
    connect = 'postgresql+psycopg2://postgres:password@localhost:5432/yelp_2'
    engine = create_engine(connect)

    chunksize = 500000
    table_name = 'all_features'
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

        data = EDA_Prep(chunk)
        data.run_all()

        data.df.to_sql('features_and_targets', con=engine,
                       index=False, if_exists='append')
        current_chunk_num += 1
        print(f'Chunk Number: {current_chunk_num} of {total_chunks} - '
              f'Loaded to table features_and_targets.')
    print('Save to Postgres Complete')


def data_prep_in_chunks_pandas():
    """
    WARNING: DO NOT USE!
    Unknown issue with pandas chunksize.
    Runs all_features table through
    EDA_Prep pipeline in chunks.
    Provides a progress printout.
    """
    connect = 'postgresql+psycopg2://postgres:password@localhost:5432/yelp_2'
    engine = create_engine(connect)

    chunksize = 10000
    table_name = 'all_features'
    row_count = int(pd.read_sql(f'SELECT COUNT(*) FROM {table_name}',
                                engine).values)
    total_chunks = int(row_count / chunksize) + 1
    current_chunk_num = 0

    query = f'''
            SELECT *
            FROM {table_name}
            ;
            '''
    for chunk in pd.read_sql(query, engine, chunksize=chunksize):
        data = EDA_Prep(chunk)
        data.run_all()
        data.df.to_sql('features_and_targets', con=engine,
                       index=False, if_exists='append')
        current_chunk_num += 1
        print(f'Chunk Number: {current_chunk_num} of {total_chunks} - '
              f'Loaded to table features_and_targets.')
    print('Save to Postgres Complete')


if __name__ == "__main__":
    data_prep_in_chunks_sql()
