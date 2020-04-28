# -*- coding: utf-8 -*-
from arekit.contrib.bert.formatters.sample.base import BaseSampleFormatter


class QaMSampleFormatter(BaseSampleFormatter):
    """
    Default, based on COLA, but includes an extra text_b.
        text_b: Question w/o S.P (S.P -- sentiment polarity)

    Notation were taken from paper:
    https://www.aclweb.org/anthology/N19-1035.pdf

    [id, label, type, text_a, text_b] -- for train
    [id, text_a, text_b] -- for test
    """

    TEXT_B = "text_b"

    __text_template = u"Что вы думаете по поводу отношения {subject} к {object} в контексте : {context} ?"

    def __init__(self, data_type):
        super(QaMSampleFormatter, self).__init__(data_type=data_type)

    def get_columns_list_with_types(self):
        dtypes_list = super(QaMSampleFormatter, self).get_columns_list_with_types()
        dtypes_list.append((self.TEXT_B, 'float64'))
        return dtypes_list

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
                continue

            yield term

    def create_row(self, parsed_news, linked_text_opinions, index_in_linked, sentence_terms):
        row = super(QaMSampleFormatter, self).create_row(
            parsed_news=parsed_news,
            linked_text_opinions=linked_text_opinions,
            index_in_linked=index_in_linked,
            sentence_terms=sentence_terms)

        inner_context = list(self.__get_inner_part_of_text(row[self.TEXT_A]))
        row[self.TEXT_B] = self.__text_template.format(
            subject=self.SUBJECT,
            object=self.OBJECT,
            context=u" ".join(inner_context))
