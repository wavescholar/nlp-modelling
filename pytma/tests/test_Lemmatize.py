import unittest

from pytma import Utility

from pytma.Lemmatize import Lemmatize
from pytma.Utility import log


class TestLemmatize(unittest.TestCase):
    def test_Lemmatize(self):
        # Get the nltk data we need
        datasets = ['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet']
        Utility.nltk_init(datasets)


        test_text = "This is the test text. Documents made up of words and/or phrases. \
            The model consists of two tables; the first table is the probability of selecting  \
            a particular word in the corpus when sampling from a particular topic, and the second \
            table is the probability of selecting a particular topic when sampling from a particular document."

        lem = Lemmatize()

        result = lem.lemmatize_nltk_with_POS(test_text)
        log.info(result)

        result = lem.lemmatize_spacy(test_text)
        log.info(result)

        result = lem.porter_stemmer(test_text)
        log.info(result)

        self.assertEqual(result[0:10],'thi is the')



if __name__ == '__main__':
    unittest.main()
