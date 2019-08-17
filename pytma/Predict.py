
##Text Classification Model using TF-IDF.
#First, import the MultinomialNB module and create a Multinomial Naive Bayes classifier object using MultinomialNB() function.

from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # This will be the unit test

    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data['Sentiment'], test_size=0.3, random_state=1)

    # Model Generation Using Multinomial Naive Bayes
    clf = MultinomialNB().fit(X_train, y_train)
    predicted= clf.predict(X_test)
    print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))