import spacy
from spacy import displacy
from nltk.parse import RecursiveDescentParser
import nltk.parse

test_text = u"It was now reading the sign that said Privet Drive — no, looking at the sign; " \
          "cats couldn't read maps or signs.He didn't see the owls swooping past in broad daylight, " \
          "though people down in the street did; they pointed and gazed open-mouthed as owl after " \
          "owl sped overhead"


nlp = spacy.load("en_core_web_sm")
doc = nlp(test_text)
displacy.serve(doc, style="dep")

rd = RecursiveDescentParser(grammar)

nltk.parse.chart.demo(2, print_times=False, trace=1,sent='I saw a dog', numparses=1)