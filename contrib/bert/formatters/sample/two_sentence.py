# -*- coding: utf-8 -*-
from arekit.contrib.bert.formatters.sample.base import BaseSampleFormatter


class TwoSentenceSampleFormatter(BaseSampleFormatter):
    """
    Provides additionally text_b parameter
    """

    TEXT_B = "text_b"

    def __init__(self, data_type):
        super(TwoSentenceSampleFormatter, self).__init__(data_type=data_type)

    def get_text_template(self):
        raise NotImplementedError()

    def get_columns_list_with_types(self):
        dtypes_list = super(TwoSentenceSampleFormatter, self).get_columns_list_with_types()
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
        row = super(TwoSentenceSampleFormatter, self).create_row(
            parsed_news=parsed_news,
            linked_text_opinions=linked_text_opinions,
            index_in_linked=index_in_linked,
            sentence_terms=sentence_terms)

        inner_context = list(self.__get_inner_part_of_text(row[self.TEXT_A]))

        text_template = self.get_text_template()
        row[self.TEXT_B] = text_template.format(
            subject=self.SUBJECT,
            object=self.OBJECT,
            context=u" ".join(inner_context))
