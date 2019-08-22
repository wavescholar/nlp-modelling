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

    def naiveBayesSentiment(self, text):
        """
        Supervised sentiment analysis.

        :param text:
        :return:
        """
        dictionary = set(word.lower() for passage in text for word in word_tokenize(passage[0]))

        t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in text]

        classifier = nltk.NaiveBayesClassifier.train(t)

        test_data = "Manchurian was hot and spicy"
        test_data_features = {word.lower(): (word in word_tokenize(test_data.lower())) for word in dictionary}

        log.info(classifier.classify(test_data_features))

    def valenceSentiment(self,text):
        """
        Unsupervised sentiment analysis.

        :param text:
        :return:
        """
        sid = SentimentIntensityAnalyzer()
        for sentence in text:
            log.info(sentence)
            ss = sid.polarity_scores(sentence)
            for k in ss:
                log.info('{0}: {1}, '.format(k, ss[k]), end ='')
                log.info()

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

    sent.naiveBayesSentiment(test_text_supervised)

    sent.valenceSentiment(test_text_unsupervised)

    log.info("done")

    # # Get the movie sentiment data and implement in test
    # X_train, X_test, y_train, y_test = train_test_split(text_tf, data['Sentiment'], test_size=0.3, random_state=123)
    # from sklearn.naive_bayes import MultinomialNB
    # from sklearn import metrics
    #
    # # Model Generation Using Multinomial Naive Bayes
    # clf = MultinomialNB().fit(X_train, y_train)
    # predicted = clf.predict(X_test)
    # log.info("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))