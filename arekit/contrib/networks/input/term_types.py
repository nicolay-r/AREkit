class TermTypes(object):
    """ Types of input terms that may occur within the
        input sequence of the neural network moodels.
    """

    WORD = "word"
    ENTITY = "entity"
    FRAME = "frame"
    TOKEN = "token"

    @staticmethod
    def iter_types():
        return [TermTypes.WORD, TermTypes.ENTITY, TermTypes.FRAME, TermTypes.TOKEN]
