import nltk

def get_nltk_POS(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    pos_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return pos_dict.get(tag, wordnet.NOUN)
