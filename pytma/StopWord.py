#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.feature_extraction import stop_words
from nltk.corpus import stopwords
import spacy
from enum import Enum
import nltk

from pytma.Utility import log


class StopWord:
    class SwLibEnum(Enum):
        scikit_sw = 1
        nltk_sw = 2
        spacy_sw = 3

    def __init__(self, text):
        self.spacy_nlp = spacy.load('en_core_web_sm')
        self.text = text
        self.swtype = self.SwLibEnum.spacy_sw

    @staticmethod
    def scikit_english():
        return stop_words.ENGLISH_STOP_WORDS

    @staticmethod
    def nltk_english():
        return set(stopwords.words('english'))


    @staticmethod
    def spacy_english():
        return spacy.lang.en.stop_words.STOP_WORDS

    def remove(self, stopwordlib=SwLibEnum.spacy_sw):
        self.swtype = stopwordlib
        if self.swtype == StopWord.SwLibEnum.scikit_sw:
            stop_words = self.scikit_english()
        if self.swtype == StopWord.SwLibEnum.nltk_sw:
            stop_words = self.nltk_english()
        if self.swtype == StopWord.SwLibEnum.spacy_sw:
            stop_words = self.spacy_english()

        to_lower = [w.lower() if not w.isupper() else w for w in self.text]
        stopword_processed = [w for w in to_lower if not w.lower() in stop_words and len(w) > 1]

        return stopword_processed

if __name__ == '__main__':

    # This will be the unit test
    test_text = "He received multiple nominations for Nobel Prize in Literature every year from 1902 to 1906, \
     and nominations for Nobel Peace Prize in 1901, 1902 and 1910, and his miss of the prize is a major Nobel \
     prize controversy.[3][4][5][6] Born to an aristocratic Russian family in 1828,[2] he is best known for \
      the novels War and Peace (1869) and Anna Karenina (1877),[7] often cited as pinnacles of realist fiction. \
    [2] He first achieved literary acclaim in his twenties with his semi-autobiographical trilogy,  \
    Childhood, Boyhood, and Youth (1852–1856), and Sevastopol Sketches (1855), based upon his experiences \
    in the Crimean War."

    token = nltk.RegexpTokenizer(r'[a-zA-Z]+')
    word_tokens = token.tokenize(test_text)
    sw = StopWord(word_tokens)

    swList = StopWord.SwLibEnum.spacy_sw
    test_text_sw_removed_spacy = sw.remove(swList)
    log.info(test_text_sw_removed_spacy)

    swList = StopWord.SwLibEnum.scikit_sw
    test_text_sw_removed_scikit = sw.remove(swList)
    log.info(test_text_sw_removed_scikit)

    swList = StopWord.SwLibEnum.nltk_sw
    test_text_sw_removed_nltk = sw.remove(swList)
    log.info(test_text_sw_removed_nltk)

    log.info("done")