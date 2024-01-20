### Preprocessing part ###


## Libraries ##


import pandas as pd
from nltk.corpus import stopwords
from nltk.data import load


## Functions ##


def clean_df(df):

    df_clean = df.drop('Unnamed: 0',axis=1)
    df_clean = df_clean.dropna(subset=['Review']).reset_index(drop=True)

    return df_clean

def drop_stop_words(corpus):

    if type(corpus) == pd.core.frame.DataFrame:
        corpus = corpus['Review'].to_list()

    stops = set(stopwords.words('french'))

    clean_corpus = []
    for doc in corpus:
        new_doc = ""
        for word in doc.split(' '):
            if not word in stops:
                new_doc = new_doc + " " + word
        clean_corpus.append(new_doc[1:])

    return clean_corpus

def tokenize(corpus):


