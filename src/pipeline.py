
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2


# Helper functions for counting and finding max.
# TODO: Combine or remove and replace with simpler solution.
def count_friends(x):
    friend_count = 0
    if x in ['None', None, '']:
        friend_count = 0
    else:
        y = x.split(',')
        friend_count = len(y)
    return friend_count


def count_elite(x):
    elite_count = 0
    if x in ['None', None, '']:
        elite_count = 0
    else:
        y = x.split(',')
        elite_count = len(y)
    return elite_count


def count_checkins(x):
    checkin_count = 0
    if x in ['None', None, '']:
        checkin_count = 0
    else:
        y = x.split(',')
        checkin_count = len(y)
    return checkin_count


def most_recent_elite(x):
    most_recent = 0
    if x in ['None', None, '']:
        most_recent = 0
    else:
        y = x.split(',')
        most_recent = max(y)
    return most_recent


if __name__ == "__main__":
    # TODO: Figure out batch loading into pandas.
    # TODO: OOP Functions and Class
    # Connection to Postgres Yelp Database - Table: restaurant_reviews_final
    conn = psycopg2.connect(database="yelp", user="postgres",
                            password='password', host="localhost", port="5432")
    query = '''
            SELECT *
            FROM restaurant_reviews_final
            LIMIT 10000
            ;
            '''
    df = pd.read_sql(query, conn)

    # TODO: Do I need to do this?
    test = df.copy()

    # Set index to review_id (unique identifier for each review)
    test.set_index('review_id', inplace=True, verify_integrity=True)

    # Delete unneeded feature columns.
    columns_to_drop = ['restaurant_name', 'restaurant_address', 'user_name',
                       'restaurant_city', 'restaurant_state',
                       'restaurant_postal_code', 'user_id', 'business_id',
                       'restaurant_categories']
    test.drop(labels=columns_to_drop, axis=1, inplace=True)

    # Sum columns where more useful than individual columns.
    review_positive_columns = ['review_useful', 'review_funny', 'review_cool']
    user_positive_columns = ['user_useful', 'user_funny', 'user_cool']
    compliment_columns = ['user_compliment_hot', 'user_compliment_more',
                          'user_compliment_profile', 'user_compliment_cute',
                          'user_compliment_list', 'user_compliment_note',
                          'user_compliment_plain', 'user_compliment_cool',
                          'user_compliment_funny', 'user_compliment_writer',
                          'user_compliment_photos']
    test['review_total_positives'] = test[review_positive_columns].sum(axis=1)
    test['user_total_positives'] = test[user_positive_columns].sum(axis=1)
    test['user_total_compliments'] = test[compliment_columns].sum(axis=1)
    test.drop(labels=review_positive_columns, axis=1, inplace=True)
    test.drop(labels=user_positive_columns, axis=1, inplace=True)
    test.drop(labels=compliment_columns, axis=1, inplace=True)

    # Split and count list like columns.
    test['user_friends_count'] = test.user_friends.apply(count_friends)
    test['user_elite_count'] = test.user_elite.apply(count_elite)
    test['restaurant_checkin_count'] = test.restaurant_checkins.apply(count_checkins)
    test.drop(labels=['user_friends', 'restaurant_checkins'], axis=1, inplace=True)

    # Years since last user elite.
    # TODO: Simplify and fold into other functions.
    test['user_elite_most_recent'] = test.user_elite.apply(most_recent_elite)
    test['user_most_recent_elite'] = pd.to_numeric(test['user_elite_most_recent'])
    test['user_years_since_last_elite'] = 2020 - test['user_most_recent_elite']
    delete_elite = ['user_elite', 'user_elite_most_recent',
                    'user_most_recent_elite']
    test.drop(labels=delete_elite, inplace=True, axis=1)
    
    # Drop Nan/Null Values
    # Restaurant Price Range only column with Nan/Null values.
    # Only a small percentage were Nan/Null so dropping all rows with the Nan/Nulls
    # Considered filling the Nan/Nulls but determined it wasn't worth the effort.
    # Other Nan/Nulls avoided during sql inner joins and counts of columns done above.
    test.dropna(inplace=True)

    # Convert Data Types
    test['user_yelp_start'] = pd.to_datetime(test['user_yelping_since'])
    test['restaurant_price'] = pd.to_numeric(test['restaurant_price_range'])
    test.drop(labels=['user_yelping_since', 'restaurant_price_range'], axis=1, inplace=True)

    # Split off review text.
    review_text = test['review_text']
    test.drop(labels=['review_text'], axis=1, inplace=True)

    # Basic feature engineering.
    # Difference between star counts.
    # review_stars - user_average_stars_given
    test['review_stars_v_user_average'] = test['review_stars'] - test['user_average_stars_given']
    # review_stars - restaurant_overall_stars
    test['review_stars_v_restaurant_average'] = test['review_stars'] - test['restaurant_overall_stars']
    test['user_time_active_at_review_time'] = test['review_date'] - test['user_yelp_start']

    # Accounting for time.
    # TODO: Adjust columns to be relative to time spans.
    # TODO: Knowable time impacts and adjustments.
    # TODO: Unknowable time impacts and adjustments.
    database_creation = '2020-3-25 19:13:01'

    # Advanced feature engineering.
    # TODO: Creative Composite Features

    # Organize Data
    # TODO: Make sure target and features are easily distinguishable and seperate.
    # TODO: Proof for possible data leakage points.
    # TODO: Order data by source.

    # Future Work:
    # Connectivity Factor from user_friends
    # Extended time based adjustments.

    # Save finished dataframe to Postgres.
    # TODO:
    