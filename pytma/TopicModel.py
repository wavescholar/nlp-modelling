#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pyLDAvis
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.models import Phrases
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from numpy import array
from pytma.DataSources import get_transcription_data
from pytma.Utility import log
import pyLDAvis
import pyLDAvis.gensim as gensimvis

class LDAAnalysis:
    def __init__(self, docs, num_topics=5, chunksize=500, passes=20, iterations=400, eval_every=1):
        self.docs = docs
        self.lda_model = None
        self.corpus = None
        self.dictionary = None
        # Set parameters.
        self.num_topics = num_topics
        self.chunksize = chunksize
        self.passes = passes
        self.iterations = iterations
        self.eval_every = eval_every

    def docs_preprocessor(self):
        tokenizer = RegexpTokenizer(r'\w+')

        try:

            for idx in range(len(self.docs)):
                self.docs[idx] = self.docs[idx].lower()  # Convert to lowercase.
                self.docs[idx] = tokenizer.tokenize(self.docs[idx])  # Split into words.
        except ValueError:
            print("{} {}".format(idx, self.docs[idx]))

        # Remove numbers, but not words that contain numbers.
        self.docs = [[token for token in doc if not token.isdigit()] for doc in self.docs]

        # Remove words that are only one character.
        self.docs = [[token for token in doc if len(token) > 3] for doc in self.docs]

        # Lemmatize all words in documents.
        lemmatizer = WordNetLemmatizer()
        self.docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in self.docs]

    def fit(self,dictionary_filter_extremes_no_below=10, dictionary_filter_extremes_no_above=0.2):
        # Perform function on our document
        self.docs_preprocessor()
        # Create Bigram & Trigram Models

        # Add bigrams and trigrams to docs,minimum count 10 means only that appear 10 times or more.
        bigram = Phrases(self.docs, min_count=10)
        trigram = Phrases(bigram[self.docs])

        for idx in range(len(self.docs)):
            for token in bigram[self.docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    self.docs[idx].append(token)
            for token in trigram[self.docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    self.docs[idx].append(token)
        # Remove rare & common tokens
        # Create a dictionary representation of the documents.
        self.dictionary = Dictionary(self.docs)
        self.dictionary.filter_extremes(no_below=dictionary_filter_extremes_no_below, no_above=dictionary_filter_extremes_no_above)
        # Create dictionary and corpus required for Topic Modeling
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.docs]
        log.info('Number of unique tokens: %d' % len(self.dictionary))
        log.info('Number of documents: %d' % len(self.corpus))
        log.info(self.corpus[:1])

        # Make a index to word dictionary.
        temp = self.dictionary[0]  # only to "load" the dictionary.

        id2word = self.dictionary.id2token

        lda_model = LdaModel(corpus=self.corpus, id2word=id2word, chunksize=self.chunksize, \
                             alpha='auto', eta='auto', \
                             iterations=self.iterations, num_topics=self.num_topics, \
                             passes=self.passes, eval_every=self.eval_every)
        # Print the Keyword in the 5 topics
        log.info(lda_model.print_topics())
        self.lda_model = lda_model
        return self.lda_model

    # Topic Coherence
    # Each topic consists of words and n-grams, and the topic coherence is applied to the top N words from the topic.
    # It is defined as the average / median of the pairwise word-similarity scores of the words in the topic
    # (e.g. PMI). A good model will generate coherent topics, i.e., topics with high topic coherence scores.
    def topic_coherence(self):
        if self.lda_model == None:
            self.fit()

        # Compute Coherence Score using c_v
        coherence_model_lda = CoherenceModel(model=self.lda_model, texts=self.docs, dictionary=self.dictionary,
                                             coherence='c_v')
        coherence_lda_CV = coherence_model_lda.get_coherence()
        log.info('\nCoherence Score CV method: ', coherence_lda_CV)

        # Compute Coherence Score using UMass
        coherence_model_lda = CoherenceModel(model=self.lda_model, texts=self.docs, dictionary=self.dictionary,
                                             coherence="u_mass")
        coherence_lda_umass = coherence_model_lda.get_coherence()
        log.info('\nCoherence Score: ', coherence_lda_umass)

        return coherence_lda_CV, coherence_lda_umass

    def compute_coherence_values(self, limit=40, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """

        coherence_values = []
        model_list = []
        id2word = self.dictionary.id2token
        for num_topics in range(start, limit, step):
            model = LdaModel(corpus=self.corpus, id2word=id2word, num_topics=num_topics)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=self.docs, dictionary=self.dictionary,
                                            coherence='u_mass')
            log.info(coherencemodel.get_coherence())
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    def coherence_values(self, limit=40, step=6, start=2):

        model_list, coherence_values = self.compute_coherence_values(start=start, limit=limit, step=step)

        x = range(start, limit, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()


class NNMFTopicAnalysis:

    def __init__(self, docs, num_samples=400, num_features=1000, num_topics=10, top_words=20):
        self.samples = num_samples
        self.num_features = num_features
        self.num_topics = num_topics
        self.top_words = top_words
        self.docs = docs

    def fit(self):
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=self.num_features, stop_words='english')
        tfidf = vectorizer.fit_transform(self.docs[:self.samples])

        nmf = NMF(n_components=self.num_topics, random_state=1).fit(tfidf)

        feature_names = vectorizer.get_feature_names()

        for topic_idx, topic in enumerate(nmf.components_):
            log.info("Topic #%d:" % topic_idx)
            log.info(" ".join([feature_names[i]
                               for i in topic.argsort()[:-self.top_words - 1:-1]]))


if __name__ == '__main__':
    # This will be the unit test

    medical_df = get_transcription_data()
    print(type(medical_df))
    text = medical_df['transcription']
    print(type(text))
    docs = array(text)
    print(type(docs))
    # =============================
    # LDA
    lda = LDAAnalysis(docs)

    do_process = True

    if do_process:
        lda.fit()
        pickle_LDAAnalysis = open("data/cache/LDAAnalysis.pkl", "wb")
        pickle.dump(lda, pickle_LDAAnalysis)
        pickle_LDAAnalysis.close()
    else:
        LDAAnalysis = pickle.load("data/cache/LDAAnalysis.pkl")

    lda.coherence_values()


    lda_vis = gensimvis.prepare(lda, lda.corpus, lda.dictionary)
    pyLDAvis.display(lda_vis)

    # =============================
    # NNMF
    nnmf = NNMFTopicAnalysis(docs=docs)
    nnmf.fit()

    print('Done')
    