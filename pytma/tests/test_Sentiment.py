import unittest

from pytma.Sentiment import Sentiment
from pytma.Utility import log


class TestSentiment(unittest.TestCase):
    def test_sentiment(self):
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

        sent = Sentiment()

        sent.naiveBayesSentimentFit(test_text_supervised)

        test_data = "Manchurian was hot and spicy"
        nb_sentiment = sent.naiveBayesSentimentPredict(test_data)
        self.assertEqual(nb_sentiment, 'pos')
        log.info("Logging NB sentiment " + nb_sentiment)

        polarity_scores = sent.valenceSentiment(test_text_unsupervised)
        self.assertEqual(polarity_scores[0], 'neg')

        log.info("Logging polarity scores " + " ".join(polarity_scores))
        log.info("done")




if __name__ == '__main__':
    unittest.main()

    print("done")