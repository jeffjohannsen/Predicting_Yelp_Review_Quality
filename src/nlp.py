
import re
import textstat
import spacy
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (RandomizedSearchCV)
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfTransformer)

from model_setup import ModelSetupInfo
from sklearn.metrics import (classification_report, r2_score,
                             mean_squared_error)
import pprint

pd.set_option('display.max_columns', 100)


class NLPPipeline():
    """
    Full nlp and text processing pipeline.
    Plugs into main modeling pipeline.
    """
    def __init__(self, goal, X_train, y_train, X_test, y_test,
                 show_results, use_cv):
        self.model_setup = ModelSetupInfo()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.goal = goal
        self.show_results = show_results
        self.use_cv = use_cv
        self.y_pred = None
        self.y_pred_proba = None

    def create_basic_text_features(self):
        """
        Adds character count, word count, and characters per word features.
        """
        for df in [self.X_train, self.X_test]:
            df['processed_text'] = \
                df['review_text'].apply(lambda x: re.sub(r'[^\w\s]', '',
                                                         x.lower()))
            df['words'] = \
                df['processed_text'].apply(lambda x: x.split(' '))
            df['review_char_count'] = \
                df['review_text'].apply(lambda x: len(x))
            df['review_word_count'] = df['words'].apply(lambda x: len(x))
            df['review_char_per_word'] = \
                df['review_char_count'] / df['review_word_count']
            df.drop(labels=['processed_text', 'words'], axis=1, inplace=True)

    def create_readability_features(self):
        """
        Adds readability features using textstat library.
        Numbers represent grade level needed to understand the text.
        """
        for df in [self.X_train, self.X_test]:
            df['review_readability_flesch_kincaid'] = \
                df['review_text'].apply(lambda x:
                                        textstat.flesch_kincaid_grade(x))
            df['review_readability_combined_methods'] = \
                df['review_text'].apply(lambda x:
                                        textstat.text_standard(x, float_output=True))

    def create_spacy_features(self):
        """
        Adds various features using Spacy's library and NLP models.

        Key Terms:
            pos_dict: Part of Speech
                      https://universaldependencies.org/u/pos/

            dep_list: Universal Dependency Relations
                      https://universaldependencies.org/u/dep/

            ent_list: Named Entity
                      https://spacy.io/api/annotation#named-entities
        """
        for df in [self.X_train, self.X_test]:
            nlp = spacy.load("en_core_web_sm")

            df['spacy_doc'] = df['review_text'].apply(lambda x: nlp(x))
            df['review_token_count'] = \
                df['spacy_doc'].apply(lambda x: len(x))
            df['review_token_count'] = \
                df['spacy_doc'].apply(lambda x:
                                      round(len([token for
                                                 token in x
                                                 if token.is_stop])
                                            / len(x), 5))
            df['review_perc_ent'] = \
                df['spacy_doc'].apply(lambda x:
                                      round(len([token for
                                                 token in x.ents])
                                            / len(x), 5))

            pos_list = {'ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ',
                        'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
                        'SCONJ', 'SYM', 'VERB', 'X'}
            for pos in pos_list:
                df[f'review_perc_{pos.lower()}'] = \
                    df['spacy_doc'].apply(lambda x:
                                          round(len([token for
                                                     token in x
                                                     if token.pos_ == pos])
                                                / len(x), 5))

            dep_list = ['ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent',
                        'amod', 'appos', 'attr', 'aux', 'auxpass', 'case',
                        'cc', 'ccomp', 'compound', 'conj', 'csubj',
                        'csubjpass', 'dative', 'dep', 'det', 'dobj',
                        'expl', 'intj', 'mark', 'meta', 'neg', 'nmod',
                        'npadvmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd',
                        'parataxis', 'pcomp', 'pobj', 'poss', 'preconj',
                        'predet', 'prep', 'prt', 'punct', 'quantmod',
                        'relcl', 'xcomp']
            for dep in dep_list:
                df[f'review_perc_{dep.lower()}'] = \
                    df['spacy_doc'].apply(lambda x:
                                          round(len([token for
                                                     token in x
                                                     if token.dep_ == dep])
                                                / len(x), 5))

            ent_list = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE',
                        'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG',
                        'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY',
                        'TIME', 'WORK_OF_ART']
            for ent in ent_list:
                df[f'review_perc_{ent.lower()}'] = \
                    df['spacy_doc'].apply(lambda x:
                                          round(len([y for
                                                     y in x.ents
                                                     if y.label_ == ent])
                                                / len(x), 5))
            df.drop('spacy_doc', axis=1, inplace=True)

    def create_prediction_features(self):
        count_vectorizer_params = {'preprocessor': None,
                                   'tokenizer': None,
                                   'strip_accents': 'unicode',
                                   'lowercase': True,
                                   'stop_words': 'english',
                                   'max_features': None,
                                   'ngram_range': (1, 1),
                                   'dtype': np.int64}
        if self.goal == 'cls':
            for nlp_model in ['sgd_cls', 'naive_bayes']:
                model = self.model_setup.nlp_models[nlp_model]
                vector_pipeline = \
                    Pipeline([('vectorizer',
                               CountVectorizer(**count_vectorizer_params)),
                              ('tfidf', TfidfTransformer()),
                              ('model', model())])
                if self.use_cv:
                    param_dict = \
                        {f'model__{k}': v for (k, v) in
                         self.model_setup.nlp_params_cv[nlp_model].items()}
                else:
                    param_dict = \
                        {f'model__{k}': v for (k, v) in
                         self.model_setup.nlp_params[nlp_model].items()}
                if nlp_model == 'sgd_cls':
                    scoring = None
                else:
                    scoring = self.model_setup.cls_scoring
                cv_pipeline = \
                    RandomizedSearchCV(estimator=vector_pipeline,
                                       param_distributions=param_dict,
                                       n_iter=10,
                                       scoring=scoring,
                                       n_jobs=None,
                                       cv=None,
                                       refit='Accuracy',
                                       random_state=7)
                cv_pipeline.fit(self.X_train['review_text'], self.y_train)
                if self.show_results:
                    if self.use_cv:
                        print(f'Model: {nlp_model.upper()}\n')
                        # print(pd.DataFrame(cv_pipeline.cv_results_))
                        print('\nBest Hyperparameters:')
                        pprint.pprint(cv_pipeline.best_estimator_)
                        print(f'''\nBest CV Accuracy:
                               {cv_pipeline.best_score_:.2f}\n''')
                    self.y_pred = \
                        cv_pipeline.predict(self.X_test['review_text'])
                    print('\nModel Prediction Results on Test Data')
                    baseline_acc = (self.y_test.value_counts().max()
                                    / len(self.y_test))
                    print(f'Baseline Accuracy: {baseline_acc:.2f}')
                    pprint.pprint(classification_report(self.y_test,
                                                        self.y_pred,
                                                        output_dict=True,
                                                        zero_division=0))
                for df in [self.X_train, self.X_test]:
                    if nlp_model == 'sgd_cls':
                        sgd_cls_pred = cv_pipeline.predict(df['review_text'])
                        for cls_name in cv_pipeline.best_estimator_.named_steps['model'].classes_:
                            df[f'review_{nlp_model}_pred_{cls_name}'] = [1 if cls_name == sgd_cls_pred[x] else 0 for x in range(len(sgd_cls_pred))]
                    elif nlp_model == 'naive_bayes':
                        nb_pred_proba = cv_pipeline.predict_proba(df['review_text'])
                        for idx, cls_name in enumerate(cv_pipeline.best_estimator_.named_steps['model'].classes_):
                            df[f'review_{nlp_model}_pred_{cls_name}'] = \
                                nb_pred_proba[:, idx]
                    else:
                        print('NLP model error.')
                        exit()

        elif self.goal == 'reg':
            for nlp_model in ['sgd_reg']:
                model = self.model_setup.nlp_models[nlp_model]
                vector_pipeline = \
                    Pipeline([('vectorizer',
                               CountVectorizer(**count_vectorizer_params)),
                              ('tfidf', TfidfTransformer()),
                              ('model', model())])
                if self.use_cv:
                    param_dict = \
                        {f'model__{k}': v for (k, v) in
                         self.model_setup.nlp_params_cv[nlp_model].items()}
                else:
                    param_dict = \
                        {f'model__{k}': v for (k, v) in
                         self.model_setup.nlp_params[nlp_model].items()}
                cv_pipeline = \
                    RandomizedSearchCV(estimator=vector_pipeline,
                                       param_distributions=param_dict,
                                       n_iter=10,
                                       scoring=self.model_setup.reg_scoring,
                                       n_jobs=None,
                                       cv=None,
                                       refit='R2 Score',
                                       random_state=7)
                cv_pipeline.fit(self.X_train['review_text'], self.y_train)
                if self.show_results:
                    if self.use_cv:
                        print(f'\nModel: {nlp_model.upper()}\n')
                        # print(pd.DataFrame(cv_pipeline.cv_results_))
                        print('Best Hyperparameters:')
                        pprint.pprint(cv_pipeline.best_estimator_)
                        print(f'''\nBest CV R2 Score:
                               {cv_pipeline.best_score_:.2f}\n''')
                    self.y_pred = \
                        cv_pipeline.predict(self.X_test['review_text'])
                    baseline_mean = np.mean(self.y_test)
                    baseline_mean_array = np.empty(len(self.y_test))
                    baseline_mean_array.fill(baseline_mean)
                    model_r2 = r2_score(self.y_test, self.y_pred)
                    baseline_r2 = r2_score(self.y_test, baseline_mean_array)
                    model_rmse = mean_squared_error(self.y_test, self.y_pred,
                                                    squared=False)
                    baseline_rmse = \
                        mean_squared_error(self.y_test, baseline_mean_array,
                                           squared=False)
                    print('\nModel Prediction Results on Test Data')
                    print(f'''Model R2: {model_r2:.2f}, Baseline R2:
                           {baseline_r2:.2f}, Difference:
                          {(model_r2 - baseline_r2):.2f}''')
                    print(f'''Model RMSE: {model_rmse:.2f}, Baseline RMSE:
                           {baseline_rmse:.2f}, Difference:
                           {(baseline_rmse - model_rmse):.2f}\n''')
                for df in [self.X_train, self.X_test]:
                    df[f'review_{nlp_model}_pred'] = \
                        cv_pipeline.predict(df['review_text'])
        else:
            print('NLP saved goal is invalid.')
            exit()


if __name__ == "__main__":
    pass
