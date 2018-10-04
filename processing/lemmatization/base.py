# -*- coding: utf-8 -*-


class Stemmer:
    """
    Interface
    """

    pos_unknown = u"unknown"
    pos_empty = u"empty"

    def lemmatize_to_list(self, text):
        pass

    def lemmatize_to_str(self, text, remove_new_lines=True):
        pass

    # TODO: POS should be moved from here
    def get_term_pos(self, term):
        pass

    # TODO: POS should be moved from here
    def get_terms_pos(self, terms):
        pass

    # TODO: POS should be moved from here
    def pos_to_int(self, pos):
        pass
