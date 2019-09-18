#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

from pytma.Utility import log


class Tokenizer:
    """Class for tokenization of text"""

    def __init__(self, text):
        """ Initialize a Tokenizer object.

        Parameters
        ----------
        text : string
            text to be tokenized

        """
        self.text = text

    def to_sentences(self):
        """
        Tokenize to sentences

        Parameters
        ----------
        self.text : string
            the text the object was initialized with

        Returns
        -------
        tokenized_text : list
            list of strings corresponding to sentences.
        """
        tokenized_text = sent_tokenize(self.text)
        return (tokenized_text)

    def to_words(self):
        """
        Tokenize to words

        Parameters
        ----------
        self.text : string
            the text the object was initialized with

        Returns
        -------
        tokenized_text : list
            list of strings corresponding to words.
        """

        tokenized_word = word_tokenize(self.text)
        return (tokenized_word)

    @staticmethod
    def regexTokenize(text):
        token = nltk.RegexpTokenizer(r'[a-zA-Z]+')
        if isinstance(text, str):
            word_tokens = token.tokenize(text)

        elif all(isinstance(item, str) for item in text):
            str_text = " ".join(text)
            word_tokens = token.tokenize(str_text)
        else:
            raise TypeError

        return word_tokens

    def freqs(self):
        """
        Calculate word frequencies in text

        Parameters
        ----------
        self.text : string
            the text the object was initialized with

        Returns
        -------
        fdist : dictionary
            word counts
        """
        fdist = FreqDist(word_tokenize(self.text))
        return (fdist)

    # Frequency Distribution Plot
    def plot_freqs(self):
        """
        Plot word frequencies

        Parameters
        ----------
        self.text : string
            the text the object was initialized with
        """

        fdist = self.freqs()
        fdist.plot(30, cumulative=True)
        plt.title("Cumulative Word Counts")
        plt.show()

if __name__ == '__main__':
    from pytma.tests.test_Tokenizer import TestTokenizer
    TestTokenizer.test_tokenizer()
