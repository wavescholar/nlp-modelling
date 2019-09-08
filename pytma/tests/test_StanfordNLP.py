import stanfordnlp

#This is a noop for now
def test_install():
    print(" Stanford NLP Test ")
    #stanfordnlp.download('en',resource_dir='stanfordnlp_resources',  force=True   )   # This downloads the English models for the neural pipeline
    useCUDA=False
    #BBCREVISIT : "RuntimeError: CUDA error: out of memory" - should not be getting this on Laptop
    if useCUDA:
        nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
        doc = nlp(u"Barack Obama was born in Hawaii.  He was elected president in 2008.")
        sentence = doc.sentences[0]
        sentence.print_dependencies()
        sentence.print_tokens()

        #We verified this works on a GCP GPU K80 with 52GB - not sure why it
        #would not work on laptop or even a medium size (16GB + K80) GCP instance - it really should.

        # < Token
        # index = 1;
        # words = [ < Word
        # index = 1;
        # text = Barack;
        # lemma = Barack;
        # upos = PROPN;
        # xpos = NNP;
        # feats = Number = Sing;
        # governor = 4;
        # depen
        # dency_relation = nsubj:pass >] >
        # < Token
        # index = 2;
        # words = [ < Word
        # index = 2;
        # text = Obama;
        # lemma = Obama;
        # upos = PROPN;
        # xpos = NNP;
        # feats = Number = Sing;
        # governor = 1;
        # depende
        # ncy_relation = flat >] >
        # < Token
        # index = 3;
        # words = [ < Word
        # index = 3;
        # text = was;
        # lemma = be;
        # upos = AUX;
        # xpos = VBD;
        # feats = Mood = Ind | Number = Sing | Person = 3 | Tense = P
        # ast | VerbForm = Fin;
        # governor = 4;
        # dependency_relation = aux:pass >] >
        # < Token
        # index = 4;
        # words = [ < Word
        # index = 4;
        # text = born;
        # lemma = bear;
        # upos = VERB;
        # xpos = VBN;
        # feats = Tense = Past | VerbForm = Part | Voice = Pa
        # ss;
        # governor = 0;
        # dependency_relation = root >] >
        # < Token
        # index = 5;
        # words = [ < Word
        # index = 5;
        # text = in;lemma = in;upos = ADP;
        # xpos = IN;
        # feats = _;
        # governor = 6;
        # dependency_relation = case >]
        # >
        # < Token
        # index = 6;
        # words = [ < Word
        # index = 6;
        # text = Hawaii;
        # lemma = Hawaii;
        # upos = PROPN;
        # xpos = NNP;
        # feats = Number = Sing;
        # governor = 4;
        # depen
        # dency_relation = obl >] >
        # < Token
        # index = 7;
        # words = [ < Word
        # index = 7;
        # text =.;lemma =.;upos = PUNCT;
        # xpos =.;feats = _;
        # governor = 4;
        # dependency_relation = punct >]
        # >