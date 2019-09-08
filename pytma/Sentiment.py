#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from pytma.Utility import log


class Sentiment:
    """
    Sentiment analyzer.  Supervised via NaiveBayes or unsupervised via Valence method.
    """
    def __init__(self):
        """
        Initializes class. Downloads vader lexicon.
        """
        nltk.download('vader_lexicon')

        self.vader_polarity_scores = list()

    def naiveBayesSentimentFit(self, text):
        """
        Supervised sentiment analysis - uses nltk to fit Naive Bayes .
        Call naiveBayesSentimentPredict to predict  using the latest fit model.

        :param text:
        :return:
        """
        dictionary = set(word.lower() for passage in text for word in word_tokenize(passage[0]))

        self.nb_dict = dictionary

        t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in text]

        classifier = nltk.NaiveBayesClassifier.train(t)

        self.nb_classifier = classifier

    def naiveBayesSentimentPredict(self,text):
        """
        Predicts sentiment.  Call naiveBayesSentimentFit on a corpus first otherwise
        an error will be thrown.

        :param text:
        :return predicted sentiment:
        """
        if hasattr( self,"nb_classifier") is False:
            raise AttributeError

        test_data_features = {word.lower(): (word in word_tokenize(text.lower())) for word in self.nb_dict}
        result = self.nb_classifier.classify(test_data_features)
        log.info("Sentiment NB predict : " + result)

        return result


    def valenceSentiment(self,text):
        """
        Unsupervised sentiment analysis.

        :param text:
        :return sentiment polarity scores:
        """
        sid = SentimentIntensityAnalyzer()
        for sentence in text:
            log.info(sentence)
            ss = sid.polarity_scores(sentence)
            for k in ss:
                print(k)
                print(ss[k])
                log.info('Logging Sentiment : {0}: {1}, '.format(k, ss[k]))
                self.vader_polarity_scores.append(k)
        return self.vader_polarity_scores

if __name__ == '__main__':
    #This will be the unit test


    test_text_supervised = [("Great place to be when you are in Bangalore.", "pos"),
                            ("The place was being renovated when I visited so the seating was limited.", "neg"),
                            ("Loved the ambience, loved the food", "pos"),
                            ("The food is delicious but not over the top.", "neg"),
                            ("Service - Little slow, probably because too many people.", "neg"),
                            ("The place is not easy to locate", "neg"),
                            ("Mushroom fried rice was spicy", "pos"),
                            ]

    test_text_unsupervised = ["Great place to be when you are in Bangalore.",
                              "The place was being renovated when I visited so the seating was limited.",
                              "Loved the ambience, loved the food", "The food is delicious but not over the top.",
                              "Service - Little slow, probably because too many people.",
                              "The place is not easy to locate", "Mushroom fried rice was tasty"]

    sent= Sentiment()

    sent.naiveBayesSentimentFit(test_text_supervised)

    test_data = "Manchurian was hot and spicy"
    nb_sentiment =sent.naiveBayesSentimentPredict(test_data)

    polarity_scores = sent.valenceSentiment(test_text_unsupervised)

    log.info("Logging polarity scores "+  " ".join(polarity_scores ) )
    log.info("done")
