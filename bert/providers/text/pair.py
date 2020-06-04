# -*- coding: utf-8 -*-
from arekit.bert.providers.text.single import SingleTextProvider
from arekit.common.labels.base import Label
from arekit.common.labels.str_fmt import StringLabelsFormatter


class PairTextProvider(SingleTextProvider):
    """
    Provides additionally text_b parameter

    Considered to utilize an inner part in context, between opinion participants.
    """

    TEXT_B = "text_b"

    def __init__(self, text_b_template, labels_formatter):
        """
        text_b_template: unicode
            assumes to include {subject}, {object}, and {context} in related template,
            and {label} (optional)
        """
        assert(isinstance(text_b_template, unicode))
        assert(isinstance(labels_formatter, StringLabelsFormatter))
        super(PairTextProvider, self).__init__()
        self.__text_b_template = text_b_template
        self.__labels_formatter = labels_formatter

    def get_text_template(self):
        raise NotImplementedError()

    def iter_columns(self):
        for col_name in super(PairTextProvider, self).iter_columns():
            yield col_name
        yield self.TEXT_B

    def __get_inner_part_of_text(self, terms):

        reading_inner_part = False
        for term in terms:
            if term == self.OBJECT or term == self.SUBJECT:
                if not reading_inner_part:
                    reading_inner_part = True
                else:
                    yield term
                    break

            if reading_inner_part:
                yield term

    def add_text_in_row(self, row, sentence_terms, s_ind, t_ind, expected_label):
        assert(isinstance(expected_label, Label))

        super(PairTextProvider, self).add_text_in_row(
            row=row,
            sentence_terms=sentence_terms,
            s_ind=s_ind,
            t_ind=t_ind,
            expected_label=expected_label)

        terms = row[self.TEXT_A].split(self.TERMS_SEPARATOR)
        inner_context = list(self.__get_inner_part_of_text(terms))

        row[self.TEXT_B] = self.__text_b_template.format(
            subject=self.SUBJECT,
            object=self.OBJECT,
            context=self._process_text(self.TERMS_SEPARATOR.join(inner_context)),
            label=self.__labels_formatter.label_to_str(expected_label))

        return row
