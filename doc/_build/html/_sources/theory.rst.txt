
NLP Theory
========================

Natural language processing (NLP) concerns itself with the interaction between natural human languages and computing devices. NLP is a major aspect of computational linguistics, and also falls within the realms of computer science and artificial intelligence.

Pre-processing steps
--------------------------------------------------

2. Tokenization

Tokenization is, generally, an early step in the NLP process, a step which splits longer strings of text into smaller pieces, or tokens. Larger chunks of text can be tokenized into sentences, sentences can be tokenized into words, etc. Further processing is generally performed after a piece of text has been appropriately tokenized.

3. Normalization

Before further processing, text needs to be normalized. Normalization generally refers to a series of related tasks meant to put all text on a level playing field: converting all text to the same case (upper or lower), removing punctuation, expanding contractions, converting numbers to their word equivalents, and so on. Normalization puts all words on equal footing, and allows processing to proceed uniformly.

4. Stemming

Stemming is the process of eliminating affixes (suffixed, prefixes, infixes, circumfixes) from a word in order to obtain a word stem.

running -> run

5. Lemmatization

Lemmatization is related to stemming, differing in that lemmatization is able to capture canonical forms based on a word's lemma.For example, stemming the word "better" would fail to return its citation form (another word for lemma); however, lemmatization would result in the following:
better -> good
It should be easy to see why the implementation of a stemmer would be the less difficult feat of the two. The goal of both stemming and lemmatization is to reduce inflectional forms and sometimes derivationally related forms of a word to a common base form. Stemming usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes.  Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma


A good starting point is [Charniak1997]_.

.. [Charniak1997] Statistical Language Learning Charniak E Language (1997) 73(3) 588


Topic Modeling
--------------------------------------------------

Topic modeling is about reducing a set of documents (a corpus) to a representation as a mixture of topics.  Each topic is a distribution over the set of words in the corpus. There are two main Bayesian topic models LDA (Latent Dirichlet Allocation) [BleiEtAl2005]_ and CTM (Correlated Topic Model) [BLeiEtAl2003]_  These are Bayesian models that use a variational formulation to fit the models. A limitation of LDA is the inability to model topic correlation.  One would expect that topics are not independent. The correlated topic model allows for the topic proportions to exhibit correlations.  This is achieved via a transformation of logistic normal distribution.  There are additional complications since we loose conjugacy in the prior. LDA is more widely used but we aim to incorporate CTM in this library. We currently use the implementation of gensim for LDA and an implementation of CTM from `https://github.com/lewer/gensim/tree/develop/gensim/models`

.. [BleiEtAl2005] Blei, D. M., & Lafferty, J. D. (2005). Correlated topic models. In Advances in Neural Information Processing Systems (pp. 147–154).

.. [BLeiEtAl2003] Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. Journal of Machine Learning Research, 3(4–5), 993–1022