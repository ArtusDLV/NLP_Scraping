### Models ###


# TODO Ideas = random forest with top influencing words as features


## Libraries ##


import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier


## Variables ##




## Functions ##


def tfidf(corpus, column = "Tokenized_review"):

    if type(corpus) == pd.core.frame.DataFrame:
        corpus = corpus[column].apply(lambda x: ' '.join(x)).to_list()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names_out(X)

    return X, features

def tfidf_ranking_one_summary(summary,tfidf_values):
    return True

def tfidf_ranking(summaries,documents,top_nb=5):
    
    try: summaries = summaries.to_list()
    except: pass
    try: documents = documents.to_list()
    except: pass    
    
    tfidf_values = tfidf(documents)
    results = []
    for summary in summaries:
        temp_ranking = tfidf_ranking_one_summary(summary,tfidf_values)
        results.append(temp_ranking.sort(reverse=True)[0:top_nb-1])

    ranking = pd.DataFrame({'Tokenized_review':summaries,'prediction':results})

    return ranking

def simple_model_ranking(x_train,x_test,y_train):

    clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    return pred

def random_forest_ranking(x_train,x_test,y_train):

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)

    return prediction
