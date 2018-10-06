# -*- coding: utf-8 -*-


class Stemmer:
    """
    Interface
    """

    _pos_unknown = u"unknown"
    _pos_empty = u"empty"

    def lemmatize_to_list(self, text):
        raise Exception("Not implemented")

    def lemmatize_to_str(self, text, remove_new_lines=True):
        raise Exception("Not implemented")

    # TODO: POS should be moved from here
    def get_term_pos(self, term):
        raise Exception("Not implemented")

    # TODO: POS should be moved from here
    def get_terms_pos(self, terms):
        raise Exception("Not implemented")

    # TODO: POS should be moved from here
    def pos_to_int(self, pos):
        raise Exception("Not implemented")

    def get_pos_unknown_token(self):
        return self._pos_unknown

    def get_pos_empty(self):
        return self._pos_empty
