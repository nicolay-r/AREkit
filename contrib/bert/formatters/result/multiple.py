import numpy as np
import pandas as pd

from arekit.common.labels.base import Label
from arekit.common.linked_text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.opinions.base import Opinion
from arekit.contrib.bert.formatters.opinions.base import BertOpinionsFormatter
from arekit.contrib.bert.formatters.result.base import BertResults
from arekit.contrib.bert.formatters.row_ids.multiple import MultipleIDFormatter


class BertMultipleResults(BertResults):

    def __init__(self):
        super(BertMultipleResults, self).__init__(ids_formatter=MultipleIDFormatter())

    # region protected methods

    def _get_column_header(self):
        # TODO. Depends on experiment format (2-scale, 3-scale),
        # TODO. Should be supported and declared in common/experiments.
        return [u'neut', u'pos', u'neg']

    def _calculate_label(self, row):
        labels_prob = [row[label] for label in self._get_column_header()]
        return Label.from_uint(np.argmax(labels_prob))

    # endregion

    # region private methods

    def __row_to_opinion(self, row, bert_opinions):
        assert(isinstance(bert_opinions, BertOpinionsFormatter))

        sample_row_id = row[self.ID]

        opinion_id = MultipleIDFormatter.sample_row_id_to_opinion_id(sample_row_id)
        _, source, target = bert_opinions.parse_row(opinion_id=opinion_id)

        return Opinion(source_value=source,
                       target_value=target,
                       sentiment=self._calculate_label(row))

    # endregion

    def iter_wrapped_linked_text_opinions(self, news_id, bert_opinions):
        assert(isinstance(news_id, int))
        assert(isinstance(bert_opinions, BertOpinionsFormatter))

        for linked_df in self._iter_linked_opinions_df(news_id=news_id):
            assert(isinstance(linked_df, pd.DataFrame))

            opinions = [self.__row_to_opinion(row=row, bert_opinions=bert_opinions)
                        for row in linked_df.iterrows()]

            yield LinkedTextOpinionsWrapper(linked_text_opinions=opinions)

