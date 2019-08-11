from sklearn.feature_extraction import stop_words
import nltk
import spacy


# [NQY18]	J. Nothman, H. Qin and R. Yurchak (2018).
# “Stop Word Lists in Free Open-source Software Packages”. In Proc. Workshop for NLP Open Source Software.

# stop words are words which are filtered out before or after processing of natural language data (text).
# Though "stop words" usually refers to the most common words in a language, there is no single universal
# list of stop words used by all natural language processing tools, and indeed not all tools even use such
# a list. Some tools specifically avoid removing these stop words to support phrase search.

class StopWordLists:

    #def __init__(self):
        #spacy_nlp = spacy.load('en_core_web_sm')

    @staticmethod
    def scikit_english():
        return stop_words.ENGLISH_STOP_WORDS

    @staticmethod
    def nltk_english():
        return nltk.corpus.stopwords
    @staticmethod
    def spacy_stopwords():
        return spacy.lang.en.stop_words.STOP_WORDS

