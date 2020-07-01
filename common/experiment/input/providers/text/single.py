from collections import OrderedDict
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.experiment.input.providers.text.terms_mapper import StringTextTermsMapper
from arekit.common.labels.base import Label
from arekit.common.synonyms import SynonymsCollection


class SingleTextProvider(object):

    TEXT_A = u'text_a'
    TERMS_SEPARATOR = u" "

    def __init__(self, entities_formatter, synonyms):
        assert (isinstance(entities_formatter, StringEntitiesFormatter))
        assert(isinstance(synonyms, SynonymsCollection) or synonyms is None)
        self._entities_formatter = entities_formatter
        self._synonyms = synonyms
        self._mapper = StringTextTermsMapper(entities_formatter=entities_formatter,
                                             synonyms=synonyms)

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
