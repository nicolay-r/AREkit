

class POSTagger:
    """
    Interface
    """

    _pos_unknown = u"unknown"
    _pos_empty = u"empty"

    def get_term_pos(self, term):
        raise Exception("Not implemented")

    def get_terms_pos(self, terms):
        raise Exception("Not implemented")

    def pos_to_int(self, pos):
        raise Exception("Not implemented")

    def get_pos_unknown_token(self):
        return self._pos_unknown

    def get_pos_empty(self):
        return self._pos_empty
