import nltk
from nltk.tokenize import word_tokenize

# Step 1 – Training data
text = [("Great place to be when you are in Bangalore.", "pos"),
        ("The place was being renovated when I visited so the seating was limited.", "neg"),
        ("Loved the ambience, loved the food", "pos"),
        ("The food is delicious but not over the top.", "neg"),
        ("Service - Little slow, probably because too many people.", "neg"),
        ("The place is not easy to locate", "neg"),
        ("Mushroom fried rice was spicy", "pos"),
        ]

# Step 2
dictionary = set(word.lower() for passage in text for word in word_tokenize(passage[0]))

# Step 3
t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in text]

# Step 4 – the classifier is trained with sample data
classifier = nltk.NaiveBayesClassifier.train(t)

test_data = "Manchurian was hot and spicy"
test_data_features = {word.lower(): (word in word_tokenize(test_data.lower())) for word in dictionary}

print(classifier.classify(test_data_features))

#===================================================================


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
hotel_rev = ["Great place to be when you are in Bangalore.","The place was being renovated when I visited so the seating was limited.","Loved the ambience, loved the food","The food is delicious but not over the top.","Service - Little slow, probably because too many people.","The place is not easy to locate","Mushroom fried rice was tasty"]
sid = SentimentIntensityAnalyzer()
for sentence in hotel_rev:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in ss:
        print('{0}: {1}, '.format(k, ss[k]), end ='')
        print()
