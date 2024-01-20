### Models ###


## Libraries ##


from sklearn.feature_extraction.text import TfidfVectorizer


## Variables ##




## Functions ##


def tfidf(corpus):

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names_out(X)

    return X, features