import sys
import os
import time

# import libraries

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')

import pandas as pd
import numpy as np

import pickle

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

from sklearn.utils import parallel_backend

def load_data(database_filepath):
    # load data from database
    engine = create_engine(os.path.join('sqlite:///', database_filepath))
    df = pd.read_sql_table('DisasterResponse', con=engine)
    X = df[df.columns[1]]
    y = df[df.columns[5:]]
    category_names = df.columns[5:]

    return X, y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    clean_tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word.lower() not in stop_words]

    return clean_tokens


def build_model():
    # pipeline = Pipeline([
    #         ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.75)),
    #         ('tfidf',TfidfTransformer(use_idf=True)),
    #         ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=1)))
    # ])

        #'vect__ngram_range': ((1, 1), (1, 2)),
    # parameters = {
    #     'vect__max_df': (0.5, 0.75, 1.0),
    #     'tfidf__use_idf': (True, False),
    # }

    # cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=1)


    # knn = KNeighborsClassifier()
    #

    pipeline2 = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5)),
            ('tfidf',TfidfTransformer(use_idf=True)),
            ('clf', MultiOutputClassifier(KNeighborsClassifier(n_neighbors=3)))
    ])

    parameters2 = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_neighbors': (3, 7, 11, 13 ),
    }

    # cv = GridSearchCV(pipeline2, param_grid=parameters2, verbose=3, n_jobs=1)

    # return cv #pipeline
    # return pipeline
    return pipeline2


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.values[:,], Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

def elapased_time(start_time, end_time):

    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time-elapsed --> {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # with parallel_backend('multiprocessing'):
        start_time = time.time()

        print('Building model...')
        model = build_model()

        end_time = time.time()
        elapased_time(start_time, end_time)

        start_time = time.time()
        print('Training model...')
        model.fit(X_train, Y_train)
        end_time = time.time()
        elapased_time(start_time, end_time)

        start_time = time.time()
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        end_time = time.time()
        elapased_time(start_time, end_time)

            # print('model best param = ', model.best_params_)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
