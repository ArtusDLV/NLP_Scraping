### Preprocessing part ###


## Libraries ##


import pandas as pd

import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from spacy import load as Lemmatizer
from nltk.corpus import stopwords
# from nltk.data import load


## Variables ##


link_raw_data = 'data/trustpilot_en_50_page.csv'


## Functions ##


def clean_df(df=pd.read_csv(link_raw_data)):

    df_clean = df.drop('Unnamed: 0',axis=1)
    df_clean = df_clean.dropna(subset=['Review']).reset_index(drop=True)

    return df_clean

def tokenize(corpus):

    if type(corpus) == pd.core.frame.DataFrame:
        corpus = corpus['Review'].to_list()

    regex = re.compile('([^\s\w]|_)+')
    stops = set(stopwords.words('english'))
    stemmer_output = PorterStemmer()
    lemmatizer_output = Lemmatizer("en_core_web_sm")

    new_corpus = []

    for document in corpus:
        new_document = regex.sub('',document).lower()
        new_document = lemmatizer_output(new_document)
        new_document = ' '.join([token.lemma_ for token in new_document])
        new_document = word_tokenize(new_document)
        new_document = [word for word in new_document if word not in stops] # stemmer_output.stem(word)
        new_corpus.append(new_document)

    return new_corpus

# Main function - To use
def load_data():

    df = clean_df()
    corpus = tokenize(df)
    df['Tokenized_reviews'] = corpus

    return df

