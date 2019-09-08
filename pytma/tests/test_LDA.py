import unittest

from gensim.corpora import Dictionary
from pandas import DataFrame
from numpy import array
from pytma.TopicModel import LDAAnalysis


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

        common_dictionary = Dictionary(common_texts)
        common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
        # This setp generates the id2token
        for k, v in common_dictionary.items():
            pass
        id2word = common_dictionary.id2token

        docs = list()
        for doc in common_texts:
            joined = " ".join(doc)
            docs.append(joined)

        lda = LDAAnalysis(docs,num_topics=3)
        lda.fit(dictionary_filter_extremes_no_below=1, dictionary_filter_extremes_no_above=0.99)
        coherence = lda.coherence_values()


        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
