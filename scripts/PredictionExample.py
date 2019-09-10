#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import warnings
from pytma import DataSources, Utility, TopicModel
from pytma.CTMModel import CTMModel
from pytma.Utility import log

warnings.filterwarnings('ignore')
import pandas as pd
import re
import numpy as np
from numpy import array
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib as plt

showPlots = False

datasets = ['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet']
Utility.nltk_init(datasets)

medical_df = DataSources.get_transcription_data()
medial_df = [['medical_specialty', 'transcription']]
medical_df = medical_df.dropna(axis=0, how='any').reset_index()
medical_df.head(10)
print(pd.value_counts(medical_df['medical_specialty']))

if showPlots:
    pd.value_counts(medical_df['medical_specialty']).plot(kind='bar', figsize=(15, 15), color='orange')
    plt.pyplot.show()

pd.value_counts(medical_df['medical_specialty']).describe()
# run this to initlalize the preprocessing tools
token = RegexpTokenizer(r'[a-zA-Z]+')  # not sure if numbers affect results
stop_words = set(stopwords.words("english"))
skip_words = re.compile('with|without|also|dr|ms|mrs|mr|miss')
skip_x = re.compile(r'\b([Xx]*)\b')  # remove any consecutive combinations of x and X
wordnet_lemmatizer = WordNetLemmatizer()

def lemmatize_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    pos_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return pos_dict.get(tag, wordnet.NOUN)

lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w, lemmatize_pos(w)) for w in nltk.word_tokenize(medical_df["transcription"][0])]

def preprocess(text):
    skip_words_processed = skip_words.sub("", text)
    word_tokens = token.tokenize(skip_words_processed)
    to_lower = [w.lower() if not w.isupper() else w for w in word_tokens]
    stopword_processed = [w for w in to_lower if not w.lower() in stop_words]
    lemmatized = [lemmatizer.lemmatize(w, lemmatize_pos(w)) for w in stopword_processed]
    return ' '.join(lemmatized)

do_process = False
if do_process:
    string_medical_df = []
    for doc in medical_df['transcription']:
        string_medical_df.append(preprocess(doc))

    pickle_processed = open("../pytma/data/cache/PredictionExample.preprocesed.pkl", "wb")
    pickle.dump(string_medical_df, pickle_processed)
    pickle_processed.close()
else:
    with open("../pytma/data/cache/PredictionExample.preprocesed.pkl", 'rb') as pickle_file:
        string_medical_df = pickle.load(pickle_file)

cv_unigram = CountVectorizer(lowercase=False, ngram_range=(1, 1))
medical_cv_unigram = cv_unigram.fit_transform(string_medical_df)

cv_bigram = CountVectorizer(lowercase=False, ngram_range=(1, 2))
medical_cv_bigram = cv_bigram.fit_transform(string_medical_df)

transformer = TfidfVectorizer(lowercase=False, ngram_range=(1, 1))
medical_tfidf_unigram = transformer.fit_transform(string_medical_df)

transformer = TfidfVectorizer(lowercase=False, ngram_range=(1, 2))
medical_tfidf_unigram = transformer.fit_transform(string_medical_df)

# give unique ids to medical specialty for simplification
count = 0
label_to_w = {}
w_to_label = {}
label = []
for w in medical_df['medical_specialty']:
    if w not in w_to_label:
        label_to_w[count] = w
        w_to_label[w] = count
        label.append(count)
        count += 1
    else:
        label.append(w_to_label[w])

medical_df['label'] = label
medical_df = medical_df.dropna(axis=0, how='any')

for (k, v) in w_to_label.items():
    print(k, v)


def run_classifier(corpus, clf, k):
    p = []
    r = []
    f = []
    k_fold = KFold(n_splits=k, shuffle=True)
    for i in range(k):
        train, test = next(k_fold.split(corpus))
        x_train = corpus.iloc[train]
        y_train = medical_df['label'][train]
        x_test = corpus.iloc[test]
        y_test = medical_df['label'][test]
        model = clf.fit(x_train, y_train)
        print(model)
        predict = clf.predict(x_test)
        p.append(precision_score(predict, y_test, average='weighted'))
        r.append(recall_score(predict, y_test, average='weighted'))
        f.append(f1_score(predict, y_test, average='weighted'))
    matrix = confusion_matrix(y_true=y_test, y_pred=predict)
    return np.mean(p), np.mean(r), np.mean(f)


results = pd.DataFrame(columns=['Classifier', 'Corpus type', 'mean_Precision', 'mean_Recall', 'mean_F1-score'])

m_nv = MultinomialNB()

do_process = False
if do_process:
    docs = array(medical_df['transcription'])
    lda = TopicModel.LDAAnalysis(docs=docs, num_topics=25)
    lda.fit()

    lda_feature_vecs = []
    for i in range(len(docs)):
        top_topics = lda.lda_model.get_document_topics(lda.corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(25)]
        lda_feature_vecs.append(topic_vec)

    pickle_processed = open("../pytma/data/cache/PredictionExample.lda_feature_vecs.pkl", "wb")
    pickle.dump(lda_feature_vecs, pickle_processed)
    pickle_processed.close()
else:
    with open("../pytma/data/cache/PredictionExample.lda_feature_vecs.pkl", 'rb') as pickle_file:
        lda_feature_vecs = pickle.load(pickle_file)

lda_corpus = pd.DataFrame(lda_feature_vecs)
p, r, f = run_classifier(lda_corpus, m_nv, 5)

results = results.append({'Classifier': 'MultinomialNB',
                          'Corpus type': 'LDA Term Topics',
                          'mean_Precision': p,
                          'mean_Recall': r,
                          'mean_F1-score': f}, ignore_index=True)
######################

unigram_corpus_array = medical_cv_unigram.toarray()

p, r, f = run_classifier(pd.DataFrame(unigram_corpus_array), m_nv, 5)

results = results.append({'Classifier': 'MultinomialNB',
                          'Corpus type': 'Bag of words - Unigram',
                          'mean_Precision': p,
                          'mean_Recall': r,
                          'mean_F1-score': f}, ignore_index=True)
print(results)

rfc = RandomForestClassifier()
p, r, f = run_classifier(pd.DataFrame(unigram_corpus_array), rfc, 5)
results = results.append({'Classifier': 'RandomForestClassifier',
                          'Corpus type': 'Bag of words - Unigram',
                          'mean_Precision': p,
                          'mean_Recall': r,
                          'mean_F1-score': f}, ignore_index=True)


p, r, f = run_classifier(lda_corpus, rfc, 5)
results = results.append({'Classifier': 'RandomForestClassifier',
                          'Corpus type': 'LDA Term Topics',
                          'mean_Precision': p,
                          'mean_Recall': r,
                          'mean_F1-score': f}, ignore_index=True)
log.info(results)

results.to_csv('doc_classification_results.txt', sep='\t')

print('done')

