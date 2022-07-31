from arekit.contrib.utils.processing.pos.base import POSTagger


class RussianPOSTagger(POSTagger):
    """ Provides cases support ('падежи')
    """

    def get_term_case(self, term):
        raise NotImplementedError()

