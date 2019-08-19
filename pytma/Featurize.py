# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

from pytma.StopWord import StopWord
from pytma.Tokenizer import Tokenizer


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
    # This will be the unit test
    test_text = "He received multiple nominations for Nobel Prize in Literature every year from 1902 to 1906, \
     and nominations for Nobel Peace Prize in 1901, 1902 and 1910, and his miss of the prize is a major Nobel \
     prize controversy.[3][4][5][6] Born to an aristocratic Russian family in 1828,[2] he is best known for \
      the novels War and Peace (1869) and Anna Karenina (1877),[7] often cited as pinnacles of realist fiction. \
    [2] He first achieved literary acclaim in his twenties with his semi-autobiographical trilogy,  \
    Childhood, Boyhood, and Youth (1852–1856), and Sevastopol Sketches (1855), based upon his experiences \
    in the Crimean War."

    text = Tokenizer.regexTokenize(test_text)

    sw = StopWord(text)
    swList = StopWord.SwLibEnum.spacy_sw

    text = sw.remove(swList)

    print(text)

    text = " ".join(text)

    feat = Featurize(text)

    text_tf = feat.tf()

    print(text_tf)

    print(len(test_text.split()))

    print(len(set(test_text.split())))

    text_tf_idf = feat.tf_idf()

    print(text_tf_idf)

    text_vectors = feat.wtv_spacy()

    feat.pca_wv(text_vectors)

    print("done")
