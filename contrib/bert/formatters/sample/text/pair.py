# -*- coding: utf-8 -*-
from arekit.contrib.bert.formatters.sample.text.single import SingleTextProvider


class PairTextProvider(SingleTextProvider):
    """
    Provides additionally text_b parameter

    Considered to utilize an inner part in context, between opinion participants.
    """

    TEXT_B = "text_b"

    def __init__(self, text_b_template):
        """
        text_b_template: unicode
            assumes to include {subject}, {object}, and {context} in related template,
            and {label} (optional)
        """
        assert(isinstance(text_b_template, unicode))
        super(PairTextProvider, self).__init__()
        self.__text_b_template = text_b_template

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

    def add_text_in_row(self, row, sentence_terms, s_ind, t_ind):
        super(PairTextProvider, self).add_text_in_row(
            row=row,
            sentence_terms=sentence_terms,
            s_ind=s_ind,
            t_ind=t_ind)

        terms = row[self.TEXT_A].split(self.TERMS_SEPARATOR)
        inner_context = list(self.__get_inner_part_of_text(terms))

        row[self.TEXT_B] = self.__text_b_template.format(
            subject=self.SUBJECT,
            object=self.OBJECT,
            context=self.TERMS_SEPARATOR.join(inner_context))

        return row
