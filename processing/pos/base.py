class POSTagger:

    Unknown = u"unknown"
    Empty = u"empty"

    def get_term_pos(self, term):
        raise Exception("Not implemented")

    def get_terms_pos(self, terms):
        raise Exception("Not implemented")

    def pos_to_int(self, pos):
        raise Exception("Not implemented")
