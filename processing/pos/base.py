class POSTagger:

    Unknown = u"UNKNOWN"
    Empty = u"EMPTY"

    @property
    def POSCount(self):
        raise NotImplementedError()

    def get_term_pos(self, term):
        raise NotImplementedError()

    def get_terms_pos(self, terms):
        raise NotImplementedError()

    def get_term_number(self, term):
        raise NotImplementedError()

    def pos_to_int(self, pos):
        raise NotImplementedError()

