from collections.abc import Iterable

from arekit.common.context.token import Token
from arekit.common.frames.text_variant import TextFrameVariant


class TextTermsMapper(object):

    def __init__(self, is_entity_func):
        assert(callable(is_entity_func))
        self.__is_entity_func = is_entity_func

    def iter_mapped(self, terms):
        """ Performs mapping operation of each terms in a sequence
        """
        assert(isinstance(terms, Iterable))

        self._before_mapping()

        for i, term in enumerate(terms):

            if isinstance(term, str):
                m_term = self.map_word(i, term)
            elif isinstance(term, Token):
                m_term = self.map_token(i, term)
            elif isinstance(term, TextFrameVariant):
                m_term = self.map_text_frame_variant(i, term)
            elif self.__is_entity_func(term):
                m_term = self.map_entity(i, term)
            else:
                raise Exception("Unsupported type {}".format(term))

            if m_term is not None:
                yield m_term

        self._after_mapping()

    def _before_mapping(self):
        pass

    def _after_mapping(self):
        pass

    def map_word(self, w_ind, word):
        raise NotImplementedError()

    def map_token(self, t_ind, token):
        raise NotImplementedError()

    def map_text_frame_variant(self, fv_ind, text_frame_variant):
        raise NotImplementedError()

    def map_entity(self, e_ind, entity):
        raise NotImplementedError()
