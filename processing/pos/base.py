

def POSTagger:
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