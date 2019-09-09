import unittest

from gensim.corpora import Dictionary
from pandas import DataFrame
from numpy import array
from pytma import Utility

from pytma.TopicModel import LDAAnalysis, NNMFTopicAnalysis


class TestLDA(unittest.TestCase):
    def __init__(self):

        self.common_texts = [
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
    def test_lda(self):
        # Get the nltk data we need
        datasets = ['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet']
        Utility.nltk_init(datasets)

        docs = list()
        for doc in self.common_texts:
            joined = " ".join(doc)
            docs.append(joined)

        lda = LDAAnalysis(docs,num_topics=3)
        lda.fit(dictionary_filter_extremes_no_below=1, dictionary_filter_extremes_no_above=0.99)
        lda.compute_coherence_values()
        coherence = lda.coherence_values()

        self.assertEqual(True, True)

    def test_NNMF(self):
        docs = list()
        for doc in self.common_texts:
            joined = " ".join(doc)
            docs.append(joined)

        nnmf = NNMFTopicAnalysis(docs=docs)
        nnmf.fit()

if __name__ == '__main__':
    unittest.main()
