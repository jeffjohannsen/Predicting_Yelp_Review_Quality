

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (RandomizedSearchCV,
                                     train_test_split)
from sklearn.feature_extraction.text import (CountVectorizer,
                                             HashingVectorizer,
                                             TfidfTransformer,
                                             TfidfVectorizer)

from business_checkins import load_dataframe_from_yelp_2
from model_setup import ModelSetupInfo

if __name__ == "__main__":
    model_setup = ModelSetupInfo()
    
    # Lots of options for vectorizing. Spacy and NLTK expand the options here.
    # Default preprocessor, tokenizer, n-grams
    # CountVectorizer - https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    # HashingVectorizer - https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
    count_vectorizer_params = {'strip_accents': None,  # 'unicode'
                               'lowercase': True,  # False
                               'stop_words': 'english',  # others found online or made myself
                               'min_df': 1,  # 0 to 1
                               'max_df': 1,  # 0 to 1
                               'max_features': None,  # 1+
                               'ngram_range': (1, 1)}
    hashing_vectorizer_params = {'strip_accents': None,  # 'unicode'
                                 'lowercase': True,  # False
                                 'stop_words': 'english',  # others found online or made myself
                                 'min_df': 1,  # 0 to 1
                                 'max_df': 1,  # 0 to 1
                                 'n_features': (2 ** 20),
                                 'ngram_range': (1, 1),
                                 'norm': 'l2'}  # 'l1'
    vectorizer = [CountVectorizer(), HashingVectorizer()]
    
    # Tfidf
    normalizer_params = {'norm': 'l2'}  # 'l1'
    normalizer = TfidfTransformer()

    # Pipeline bringing all NLP together.
    text_pipeline = Pipeline([('vectorizer', vectorizer),
                              ('normalizer', normalizer),
                              ('model', NaiveBayesCLS)])

    