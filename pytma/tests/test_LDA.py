import unittest

from gensim.corpora import Dictionary
from pandas import DataFrame
from numpy import array
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
import pandas as pd
from pytma import Utility

from pytma.TopicModel import LDAAnalysis, NNMFTopicAnalysis


class TestLDA(unittest.TestCase):
    def test_lda(self):
        common_texts = [
            ['human', 'interface', 'computer'],
            ['survey', 'user', 'computer', 'system', 'response', 'time'],
            ['eps', 'user', 'interface', 'system'],
            ['system', 'human', 'system', 'eps'],
            ['user', 'response', 'time'],
            ['trees'],
            ['graph', 'trees'],
            ['graph', 'minors', 'trees'],
            ['graph', 'minors', 'survey']
        ]

        # Get the nltk data we need
        datasets = ['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet']
        Utility.nltk_init(datasets)

        num_topics = 3

        docs = list()
        for doc in common_texts:
            joined = " ".join(doc)
            docs.append(joined)

        lda = LDAAnalysis(docs,num_topics=num_topics)
        lda.fit(dictionary_filter_extremes_no_below=1, dictionary_filter_extremes_no_above=0.99)
        lda.compute_coherence_values()
        coherence = lda.coherence_values()

        docs = list()
        for doc in common_texts:
            joined = " ".join(doc)
            docs.append(joined)

        vectorizer = CountVectorizer(analyzer='word', max_features=5000)
        x_counts = vectorizer.fit_transform(docs)
        transformer = TfidfTransformer(smooth_idf=False)
        x_tfidf = transformer.fit_transform(x_counts)
        xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
        model = NMF(n_components=num_topics, init='nndsvd')
        model.fit(xtfidf_norm)

        def get_nmf_topics(model, n_top_words):
            # the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
            feat_names = vectorizer.get_feature_names()

            word_dict = {}
            for i in range(num_topics):
                # for each topic, obtain the largest values, and add the words they map to into the dictionary.
                words_ids = model.components_[i].argsort()[:-20 - 1:-1]
                words = [feat_names[key] for key in words_ids]
                word_dict['Topic # ' + '{:02d}'.format(i + 1)] = words

            return pd.DataFrame(word_dict)

        get_nmf_topics(model, 20)



        nnmf = NNMFTopicAnalysis(docs=docs,num_topics=3, top_words=2)
        nnmf.fit()

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
