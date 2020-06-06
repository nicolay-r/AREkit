from collections import OrderedDict
from arekit.bert.providers.text.terms_formatter import iterate_sentence_terms
from arekit.common.entities.str_mask_fmt import StringEntitiesFormatter
from arekit.common.labels.base import Label


class SingleTextProvider(object):

    TEXT_A = u'text_a'
    TERMS_SEPARATOR = u" "

    def __init__(self, entities_formatter):
        assert (isinstance(entities_formatter, StringEntitiesFormatter))
        self._entities_formatter = entities_formatter

    def iter_columns(self):
        yield SingleTextProvider.TEXT_A

    @staticmethod
    def _process_text(text):
        assert(isinstance(text, unicode))
        return text.strip()

    def add_text_in_row(self, row, sentence_terms, s_ind, t_ind, expected_label):
        assert(isinstance(row, OrderedDict))
        assert(isinstance(expected_label, Label))
        text = self.TERMS_SEPARATOR.join(iterate_sentence_terms(sentence_terms,
                                                                s_ind=s_ind,
                                                                t_ind=t_ind))
        row[self.TEXT_A] = self._process_text(text)
