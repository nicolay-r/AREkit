# -*- coding: utf-8 -*-
from arekit.processing.pos.base import POSTagger


class RussianPOSTagger(POSTagger):
    """ Provides cases support ('падежи')
    """

    def get_term_case(self, term):
        raise NotImplementedError()

