from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class Featurize():
    """
    Class for featurizing text.  BOW, tf-idf, vector embeddings
    """
    def __init__(self,text):
        self.text=text

    # Term Frequency(TF) is the count the number of words occurred in each document.
    # The main issue with Term Frequency is that it will give more weight to longer documents.
    # Term Frequency is the BoW model.
    # IDF(Inverse Document Frequency) measures the amount of information a given word provides
    # across the document. IDF is the logarithmically scaled inverse ratio of the number of
    # documents that contain the word and the total number of documents
    def tf_idf(self):
        tf=TfidfVectorizer()
        text_tf= tf.fit_transform(data['Phrase'])

if __name__ == '__main__':
    #This will be the unit test

    feat = Featurize(data)
    text_tf= feat.tf_idf()

    X_train, X_test, y_train, y_test = train_test_split(text_tf, data['Sentiment'], test_size=0.3, random_state=123)
    from sklearn.naive_bayes import MultinomialNB
    from sklearn import metrics
    # Model Generation Using Multinomial Naive Bayes
    clf = MultinomialNB().fit(X_train, y_train)
    predicted= clf.predict(X_test)
    print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))