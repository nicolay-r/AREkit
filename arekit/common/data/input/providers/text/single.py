from arekit.common.data import const
from arekit.common.data.input.terms_mapper import OpinionContainingTextTermsMapper
from arekit.common.labels.base import Label


class BaseSingleTextProvider(object):

    TEXT_A = const.TEXT
    TERMS_SEPARATOR = " "

    def __init__(self, text_terms_mapper):
        assert(isinstance(text_terms_mapper, OpinionContainingTextTermsMapper))
        self._mapper = text_terms_mapper

    def iter_columns(self):
        yield BaseSingleTextProvider.TEXT_A

    @staticmethod
    def _process_text(text):
        assert(isinstance(text, str))
        return text.strip()

    def _mapped_data_to_str(self, m_data):
        return m_data

    def _handle_mapped_data(self, m_data):
        # Optionally handle mapped data.
        pass

    def _handle_terms_and_compose_text(self, sentence_terms):
        assert(isinstance(sentence_terms, list))

        str_terms = []

        for m_data in self._mapper.iter_mapped(sentence_terms):
            str_terms.append(self._mapped_data_to_str(m_data=m_data))
            self._handle_mapped_data(m_data=m_data)

        return self.TERMS_SEPARATOR.join(str_terms)

    def add_text_in_row(self, set_text_func, sentence_terms, s_ind, t_ind, expected_label):
        assert(callable(set_text_func))
        assert(isinstance(sentence_terms, list))
        assert(isinstance(expected_label, Label))

        self._mapper.set_s_ind(s_ind)
        self._mapper.set_t_ind(t_ind)
        set_text_func(column=self.TEXT_A,
                      value=self._process_text(text=self._handle_terms_and_compose_text(sentence_terms)))
