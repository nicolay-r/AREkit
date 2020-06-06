from collections import OrderedDict

from arekit.common.entities.base import Entity
from arekit.common.entities.entity_mask import StringEntitiesFormatter
from arekit.common.entities.types import EntityType
from arekit.common.labels.base import Label
from arekit.processing.text.token import Token


class SingleTextProvider(object):

    TEXT_A = u'text_a'
    TERMS_SEPARATOR = u" "

    def __init__(self, entities_formatter):
        assert(isinstance(entities_formatter, StringEntitiesFormatter))
        self._entities_formatter = entities_formatter

    def iter_columns(self):
        yield SingleTextProvider.TEXT_A

    def _iterate_sentence_terms(self, sentence_terms, s_ind=None, t_ind=None):
        assert(isinstance(sentence_terms, list))
        assert(isinstance(s_ind, int) or s_ind is None)
        assert(isinstance(t_ind, int) or t_ind is None)

        for i, term in enumerate(sentence_terms):

            if isinstance(term, unicode):
                yield term
            elif isinstance(term, Entity):
                if i == s_ind:
                    yield self._entities_formatter.to_string(EntityType.Subject)
                elif i == t_ind:
                    yield self._entities_formatter.to_string(EntityType.Object)
                else:
                    yield self._entities_formatter.to_string(EntityType.Other)
            elif isinstance(term, Token):
                yield term.get_original_value()

    @staticmethod
    def _process_text(text):
        assert(isinstance(text, unicode))
        return text.strip()

    def add_text_in_row(self, row, sentence_terms, s_ind, t_ind, expected_label):
        assert(isinstance(row, OrderedDict))
        assert(isinstance(sentence_terms, list))
        assert(isinstance(expected_label, Label))
        text = self.TERMS_SEPARATOR.join(self._iterate_sentence_terms(sentence_terms,
                                                                      s_ind=s_ind,
                                                                      t_ind=t_ind))
        row[self.TEXT_A] = self._process_text(text)
