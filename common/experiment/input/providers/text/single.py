from collections import OrderedDict

from arekit.common.experiment.input.terms_mapper import StringTextTermsMapper
from arekit.common.labels.base import Label


class SingleTextProvider(object):

    TEXT_A = u'text_a'
    TERMS_SEPARATOR = u" "

    def __init__(self, text_terms_mapper):
        assert(isinstance(text_terms_mapper, StringTextTermsMapper))
        self._mapper = text_terms_mapper

    def iter_columns(self):
        yield SingleTextProvider.TEXT_A

    @staticmethod
    def _process_text(text):
        assert(isinstance(text, unicode))
        return text.strip()

    def add_text_in_row(self, row, sentence_terms, s_ind, t_ind, expected_label):
        assert(isinstance(row, OrderedDict))
        assert(isinstance(expected_label, Label))

        self._mapper.set_s_ind(s_ind)
        self._mapper.set_t_ind(t_ind)
        text = self.TERMS_SEPARATOR.join(self._mapper.iter_mapped(sentence_terms))

        row[self.TEXT_A] = self._process_text(text)
