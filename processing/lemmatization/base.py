# -*- coding: utf-8 -*-


class Stemmer:
    """
    Interface
    """

    def lemmatize_to_list(self, text):
        raise Exception("Not implemented")

    def lemmatize_to_str(self, text):
        raise Exception("Not implemented")

    def is_adjective(self, pos_type):
        raise Exception("Not implemented")

    def is_noun(self, pos_type):
        raise Exception("Not implemented")
