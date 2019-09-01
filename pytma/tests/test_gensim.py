from gensim import corpora, models

def test_LDA():
    texts = [['human', 'interface', 'computer'],
     ['survey', 'user', 'computer', 'system', 'response', 'time'],
     ['eps', 'user', 'interface', 'system'],
     ['system', 'human', 'system', 'eps'],
     ['user', 'response', 'time'],
     ['trees'],
     ['graph', 'trees'],
     ['graph', 'minors', 'trees'],
     ['graph', 'minors', 'survey']]

    # build the corpus, dict and train the model
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=2,
                                     random_state=0, chunksize=2, passes=10)

    # show the topics
    topics = model.show_topics()
    for topic in topics:
        print( topic )
    ### (0, u'0.159*"system" + 0.137*"user" + 0.102*"response" + 0.102*"time" + 0.099*"eps" + 0.090*"human" + 0.090*"interface" + 0.080*"computer" + 0.052*"survey" + 0.030*"minors"')
    ### (1, u'0.267*"graph" + 0.216*"minors" + 0.167*"survey" + 0.163*"trees" + 0.024*"time" + 0.024*"response" + 0.024*"eps" + 0.023*"user" + 0.023*"system" + 0.023*"computer"')

    # get_document_topics for a document with a single token 'user'
    text = ["user"]
    bow = dictionary.doc2bow(text)
    print( "get_document_topics", model.get_document_topics(bow))
    ### get_document_topics [(0, 0.74568415806946331), (1, 0.25431584193053675)]

    # get_term_topics for the token user
    print ("get_term_topics: ", model.get_term_topics("user", minimum_probability=0.000001) )
    ### get_term_topics:  [(0, 0.1124525558321441), (1, 0.006876306738765027)]