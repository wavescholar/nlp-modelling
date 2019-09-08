#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

from pytma.StopWord import StopWord
from pytma.Tokenizer import Tokenizer
from pytma.Utility import log


class Featurize:
    """
    Class for featurizing text.  BOW, tf-idf, vector embeddings
    """

    def __init__(self, text):
        self.text = text
        self.text_tokens = None

    # Term Frequency(TF) is the count the number of words occurred in each document.
    # The main issue with Term Frequency is that it will give more weight to longer documents.
    # Term Frequency is the BoW model.
    # IDF(Inverse Document Frequency) measures the amount of information a given word provides
    # across the document. IDF is the logarithmically scaled inverse ratio of the number of
    # documents that contain the word and the total number of documents
    def tf_idf(self):
        tf = TfidfVectorizer()
        text_tf = tf.fit_transform(self.text.split())
        return text_tf

    def tf(self):
        # tokenizer to remove unwanted elements from out data like symbols and numbers
        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        # stop_words : string {‘english’}, ‘english’is a built-in stop word list for English .
        # There are several known issues with ‘english’ and we should consider an alternative
        cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
        text_counts = cv.fit_transform(self.text.split())
        return text_counts

    def wtv_spacy(self):
        import numpy as np
        import spacy
        nlp = spacy.load("en")
        text_tokens = nlp(self.text)
        self.text_tokens = text_tokens
        text_vectors = np.vstack([word.vector for word in text_tokens if word.has_vector])
        return text_vectors

    def pca_wv(self, vecs):

        if self.text_tokens == None:
            self.wtv_spacy()

        pca = PCA(n_components=2)
        text_vecs_transformed = pca.fit_transform(vecs)
        text_vecs_transformed = np.c_[self.text_tokens, text_vecs_transformed]

        x = text_vecs_transformed[:, 1]
        y = text_vecs_transformed[:, 2]
        x = [float(i) for i in x]
        y = [float(i) for i in y]

        plt.scatter(x, y)
        plt.xticks(x, rotation='vertical')
        for i, txt in enumerate(self.text.split()):
            plt.annotate(txt, (x[i], y[i]))

        plt.show()


if __name__ == '__main__':
    from pytma.tests.test_Featureize import test_Featureize

    test_Featureize()