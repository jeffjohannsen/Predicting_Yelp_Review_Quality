
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer


from business_checkins import load_dataframe_from_yelp_2

if __name__ == "__main__":
    query = '''
            SELECT *
            FROM review_text_only
            LIMIT 10000
            ;
            '''
    df = load_dataframe_from_yelp_2(query)
    print(df.info())
    print(df.head(5))

    