from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

test_text = """Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
The sky is pinkish-blue. You shouldn't eat cardboard"""

class Tokenizer:
    def __init__(self, text):
        self.text = text

    def to_sentences(self):
        tokenized_text = sent_tokenize(self.text)
        return (tokenized_text)

    def to_words(self):
        tokenized_word = word_tokenize(self.text)
        return (tokenized_word)

    def freqs(self):
        fdist = FreqDist(word_tokenize(self.text))
        return (fdist)

    # Frequency Distribution Plot
    def plot_freqs(self):
        fdist = self.freqs(self)
        fdist.plot(30, cumulative=True)
        plt.title("Cumulative Word Counts")
        plt.show()


if __name__ == '__main__':
    #This will be the unit test

    t= Tokenizer(test_text)
    print(t.to_words())
    print(t.to_sentences())
    print(t.freqs())
    t.plot_freqs()
