import spacy
from spacy import displacy
from nltk.parse import RecursiveDescentParser
import nltk.parse


class ParseText:
    def __init__(self,text):
        self.nlp = spacy.load("en_core_web_sm")
        self.text = text

    def display_parse(self):
        doc = self.nlp(self.text)
        displacy.serve(doc, style="dep")

    #rd = RecursiveDescentParser(grammar)
    def chart_parse(self):
        nltk.parse.chart.demo(2, print_times=False, trace=1,sent=self.text, numparses=1)

if __name__ == '__main__':
    # This will be the unit test

    test_text = u"It was now reading the sign that said Privet Drive — no, looking at the sign; " \
              "cats couldn't read maps or signs.He didn't see the owls swooping past in broad daylight, " \
              "though people down in the street did; they pointed and gazed open-mouthed as owl after " \
              "owl sped overhead"
    pt = ParseText(test_text)

    #Needs work
    #pt.chart_parse()

    pt.display_parse()


