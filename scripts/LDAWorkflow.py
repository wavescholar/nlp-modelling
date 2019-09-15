#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import path

import pandas as pd
import re
import nltk
import gensim
from pytma import Utility
from pytma import DataSources
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import spacy
import pyLDAvis
import pyLDAvis.gensim as gensimvis
import pickle

#This script is going to go away - most of the fuctionality has been ported to pytma

if __name__ == '__main__':

    # Get the nltk data we need
    datasets = ['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet']
    Utility.nltk_init(datasets)

    medical_df = DataSources.get_transcription_data()

    # run this to initialize the pre-possessing tools
    token = RegexpTokenizer(r'[a-zA-Z]+')  # not sure if numbers affect results
    stop_words = set(stopwords.words("english"))
    skip_words = re.compile('with|without|also|dr|ms|mrs|mr|miss')
    skip_x = re.compile(r'\b([Xx]*)\b')

    def get_nltk_POS(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        pos_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return pos_dict.get(tag, wordnet.NOUN)

    def lemmatize_nltk_with_POS(text):
        lemmatizer = WordNetLemmatizer()
        result = [lemmatizer.lemmatize(w, get_nltk_POS(w)) for w in nltk.word_tokenize(text)]
        return result


    word_tokens = token.tokenize(medical_df["transcription"][0])
    stopword_processed = [w for w in word_tokens if not w in stop_words]
    lemmatized_nltk = lemmatize_nltk_with_POS(" ".join(stopword_processed))


    def lemmatize_spacy(text):
        nlp = spacy.load('en', disable=['parser', 'ner'])
        result = nlp(text)
        return result


    lemmatized_spacey = lemmatize_spacy(medical_df["transcription"][0])

    print(medical_df["transcription"][0])
    print(" ".join([token.lemma_ for token in lemmatized_spacey]))


    # Applying the pre-processing to entire transcription
    def preprocess(text):
        lemmatizer = WordNetLemmatizer()
        skip_words_processed = skip_words.sub("", text)
        skip_x_processed = skip_x.sub('', skip_words_processed)
        word_tokens = token.tokenize(skip_x_processed)
        to_lower = [w.lower() if not w.isupper() else w for w in word_tokens]
        stopword_processed = [w for w in to_lower if not w.lower() in stop_words and len(w) > 1]
        lemmatized = [lemmatizer.lemmatize(w, get_nltk_POS(w)) for w in stopword_processed]
        return lemmatized

    pickle_filename="../pytma/data/cache/LDAWorkflow.preprocesed.pkl"
    if path.exists(pickle_filename):
        do_process = False
    else:
        do_process = True

    if do_process:
        import multiprocessing

        pool = multiprocessing.Pool(multiprocessing.cpu_count())

        processed = medical_df["transcription"].pool.map(preprocess)

        pickle_processed = open(pickle_filename, "wb")
        pickle.dump(processed, pickle_processed)
        pickle_processed.close()
    else:
        with open(pickle_filename, 'rb') as pickle_file:
            processed = pickle.load(pickle_file)

    dictionary = gensim.corpora.Dictionary(processed)  # dictionary gives unique ids to the words for easier processing
    print(type(dictionary))

    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 10:
            break
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    # Bag of Words & TF-IDF model
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed]

    md_tfidf = gensim.models.TfidfModel(bow_corpus)
    # Transform the whole corpus via TfIdf and index it, in preparation for similarity queries:
    index = gensim.similarities.SparseMatrixSimilarity(md_tfidf[bow_corpus], num_features=12)

    corpus_md_tfidf = md_tfidf[bow_corpus]

    # observe the overall distribution of the lexicons over the documents
    dict_bow = {}
    for doc in bow_corpus:  # access list of bow
        for count in doc:  # access tuple
            if count[0] not in dict_bow:
                dict_bow[dictionary[count[0]]] = count[1]
            else:
                dict_bow[dictionary[count[0]]] += count[1]
    y_min = min(dict_bow, key=dict_bow.get)
    y_max = max(dict_bow, key=dict_bow.get)

    # sorted(dict1, key=dict1.get)
    dict_bow = sorted(dict_bow.items(), key=lambda x: x[1])
    print(dict_bow[0:50])
    print(dict_bow[(len(dict_bow) - 50):len(dict_bow)])

    count_lower = pd.DataFrame(dict_bow[0:50], columns=['word', 'frequency'])
    count_higher = pd.DataFrame(dict_bow[(len(dict_bow) - 50):len(dict_bow)], columns=['word', 'frequency'])

    count_lower.plot(kind='bar', x='word', figsize=(15, 15))
    count_higher.plot(kind='bar', x='word', figsize=(15, 15))
    print(y_min, y_max)


    # Running LDA
    def lda_to_dict(model, num_topics, num_words):
        word_dict = {}
        for i in range(num_topics):
            words = model.show_topic(i, topn=num_words)
            word_dict['Topic # ' + '{:02d}'.format(i + 1)] = [i[0] for i in words]
        return word_dict


    topics = 15
    words = 20

    lda_model = gensim.models.LdaMulticore(corpus_md_tfidf, num_topics=topics, id2word=dictionary, passes=2, workers=4)

    lsi_model = gensim.models.LsiModel(corpus_md_tfidf, num_topics=topics, id2word=dictionary)

    print("LDA Model:")

    for idx in range(topics):
        # Print the first 10 most representative topics
        print("Topic #%s:" % idx, lda_model.print_topic(idx, 10))

    print("=" * 20)

    print("LSI Model:")

    for idx in range(topics):
        # Print the first 10 most representative topics
        print("Topic #%s:" % idx, lsi_model.print_topic(idx, 10))

    print("=" * 20)

    pickle_lda = open("../../pytma/data/cache/lda_model_15.pkl", "wb")
    pickle.dump(lda_model, pickle_lda)
    pickle_lda.close()

    doLDAVis = False
    if doLDAVis==True:
        lda_vis = gensimvis.prepare(lda_model, bow_corpus, dictionary)
        pyLDAvis.display(lda_vis)
