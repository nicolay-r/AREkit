from collections import OrderedDict

from arekit.common.entities.base import Entity
from arekit.common.labels.base import Label
from arekit.processing.text.token import Token


class SingleTextProvider(object):

    TEXT_A = u'text_a'
    TERMS_SEPARATOR = u" "

    SUBJECT = u"X"
    OBJECT = u"Y"
    ENTITY = u"E"

    def __init__(self):
        pass

    def iter_columns(self):
        yield SingleTextProvider.TEXT_A

    @staticmethod
    def __iterate_sentence_terms(sentence_terms, s_ind, t_ind):

        for i, term in enumerate(sentence_terms):

            if isinstance(term, unicode):
                yield term
            elif isinstance(term, Entity):
                if i == s_ind:
                    yield SingleTextProvider.SUBJECT
                elif i == t_ind:
                    yield SingleTextProvider.OBJECT
                else:
                    yield SingleTextProvider.ENTITY
            elif isinstance(term, Token):
                yield term.get_meta_value()

    @staticmethod
    def _process_text(text):
        assert(isinstance(text, unicode))
        return text.strip()

    def add_text_in_row(self, row, sentence_terms, s_ind, t_ind, expected_label):
        assert(isinstance(row, OrderedDict))
        assert(isinstance(expected_label, Label))
        text = self.TERMS_SEPARATOR.join(self.__iterate_sentence_terms(sentence_terms,
                                                                       s_ind=s_ind,
                                                                       t_ind=t_ind))
        row[self.TEXT_A] = self._process_text(text)
