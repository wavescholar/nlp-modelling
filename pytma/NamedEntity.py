#!/usr/bin/env python
# -*- coding: utf-8 -*-
import spacy


class NamedEntity:
    """
    Named entity recognition
    """

    def __init__(self, text):
        """ Initialize a Tokenizer object.

        Parameters
        ----------
        text : string
            text to be tokenized

        """
        self.text = text
        nlp = spacy.load("en_core_web_sm")
        self.doc = nlp(text)

    def get_named_entities(self):
        """

        :return:
        """
        for ent in self.doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)