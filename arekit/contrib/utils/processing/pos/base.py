# TODO. Move to base class.
class POSTagger:

    def get_term_pos(self, term):
        raise NotImplementedError()

    def get_term_number(self, term):
        raise NotImplementedError()

    def pos_to_int(self, pos):
        raise NotImplementedError()

