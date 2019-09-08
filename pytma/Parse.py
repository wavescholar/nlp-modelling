#!/usr/bin/env python
# -*- coding: utf-8 -*-

import spacy
from spacy import displacy
from nltk.parse import RecursiveDescentParser
import nltk.parse
from spacy.pipeline import DependencyParser

from pytma.Utility import log


class ParseText:
    """
    Class for parsing text.  Uses spacey.
    """
    def __init__(self,text):
        self.nlp = spacy.load("en_core_web_sm")
        self.text = text

    def display_parse(self):
        """
        sets up web page with parse diagram
        :return:
        """
        doc = self.nlp(self.text)
        displacy.serve(doc, style="dep")

    def dependency_parse(self):
        """
        Dependency parser : Analyzes the grammatical structure of a sentence,
        establishing relationships between "head" words and words which modify those heads
        :return:
        """
        spacy.load("en_core_web_sm")
        parser = DependencyParser(self.nlp.vocab)
        doc = self.nlp(self.text)
        processed = parser(doc)
        log.info("Dependency Parsing : " + processed)
        return processed

    def noun_chunks(self):
        """
        Looks for n-gram noun phrases
        Think of noun chunks as a noun plus the words describing the noun –
        for example, “the lavish green grass” or “the world’s largest tech fund”
        :return:
        """
        doc = self.nlp(self.text)
        result = list()
        for chunk in doc.noun_chunks:
            result.append(chunk.text)
            log.info(chunk.text +" " +chunk.root.text +" " + chunk.root.dep_+" " +chunk.root.head.text)
        return result

    #TODO - needs work
    def chart_parse(self):
        """
        Chart parse
        :return:
        """
        nltk.parse.chart.demo(2, print_times=False, trace=1,sent=self.text, numparses=1)

if __name__ == '__main__':
    # This will be the unit test

    test_text = u"It was now reading the sign that said Privet Drive — no, looking at the sign; " \
              "cats couldn't read maps or signs.He didn't see the owls swooping past in broad daylight, " \
              "though people down in the street did; they pointed and gazed open-mouthed as owl after " \
              "owl sped overhead"

    pt = ParseText(test_text)

    noun_chunks = pt.noun_chunks()

    parsed = pt.dependency_parse()
    log.info("parse "+ parsed)

    #TODO Broken
    #pt.chart_parse()

    #Can't be run in unit test
    pt.display_parse()


