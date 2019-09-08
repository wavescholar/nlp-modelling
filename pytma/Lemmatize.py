#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer

from pytma.Utility import log


class Lemmatize:
    """Lematizer class:  Wordnet or spacy"""

    def __init__(self):
        """
        Initialize class - performs setup for Wordnet and spacy

        """
        self.lemmatizer = WordNetLemmatizer()
        self.spaceynlp = spacy.load('en', disable=['parser', 'ner'])

    def get_nltk_POS(self, word):
        """
        Get the part of speech

        :param word:
        :return:
        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        pos_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return pos_dict.get(tag, wordnet.NOUN)

    def lemmatize_nltk_with_POS(self, text):
        """
        lemmatizes with nltk - using the POS

        :param text:
        :return:
        """
        token = RegexpTokenizer(r'[a-zA-Z]+')  # not sure if numbers affect results
        stop_words = set(stopwords.words("english"))

        word_tokens = token.tokenize(text)
        stopword_processed = [w for w in word_tokens if not w in stop_words]
        text = " ".join(stopword_processed)
        result = [self.lemmatizer.lemmatize(w, self.get_nltk_POS(w)) for w in nltk.word_tokenize(text)]

        return result

    def lemmatize_spacy(self, text):
        """
        Lematizes using spacy

        :param text:
        :return:
        """
        result = self.spaceynlp(text)
        return result

    def porter_stemmer(self,text):
        stem = PorterStemmer()
        split_text = text.split()
        stemmed_words=list()
        for word in split_text:
            stemmed =stem.stem(word)
            stemmed_words.append(stemmed)
        stemmed_text = " ".join(stemmed_words)
        return stemmed_text

if __name__ == '__main__':
    # This will be the unit test

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
