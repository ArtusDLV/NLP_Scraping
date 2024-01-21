### Preprocessing part ###


## Libraries ##


import pandas as pd

import re
from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
from spacy import load as Lemmatizer
from nltk.corpus import stopwords
# from nltk.data import load

from sklearn.model_selection import train_test_split


## Variables ##


link_raw_data = 'data/trustpilot_en_50_page.csv'


## Functions ##


def clean_df(column='Review',df=pd.read_csv(link_raw_data)):

    if 'Unnamed: 0' in df.columns:
        df_clean = df.drop('Unnamed: 0',axis=1)
    else:
        df_clean = df.copy()
    df_clean = df_clean.dropna(subset=[column]).reset_index(drop=True)

    return df_clean

def tokenize(corpus,column='Review'):

    if type(corpus) == pd.core.frame.DataFrame:
        corpus = corpus[column].to_list()

    regex = re.compile('([^\s\w]|_)+')
    stops = set(stopwords.words('english'))
    # stemmer_output = PorterStemmer()
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
    df['Tokenized_review'] = corpus

    return df

def train_test(df,seed=42):

    return train_test_split(df['Tokenized_headline'], df['Tokenized_review'], test_size=0.33, random_state=seed)

