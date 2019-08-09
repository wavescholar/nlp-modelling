
import spacy
from spacy import displacy
test_text = u"It was now reading the sign that said Privet Drive — no, looking at the sign; " \
          "cats couldn't read maps or signs.He didn't see the owls swooping past in broad daylight, " \
          "though people down in the street did; they pointed and gazed open-mouthed as owl after " \
          "owl sped overhead"
nlp = spacy.load("en_core_web_sm")
doc = nlp(test_text)
displacy.serve(doc, style="dep")