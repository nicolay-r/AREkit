import pandas as pd

from arekit.contrib.bert.formatters.opinions.base import BertOpinionsFormatter
from arekit.contrib.bert.formatters.result.base import BertResults
from arekit.contrib.bert.formatters.row_ids.binary import BinaryIDFormatter


class BertBinaryResults(BertResults):

    def __init__(self):
        super(BertBinaryResults, self).__init__(ids_formatter=BinaryIDFormatter())

    def _get_column_header(self):
        return [u'yes', u'no']

    def iter_wrapped_linked_text_opinions(self, news_id, bert_opinions):
        assert(isinstance(news_id, int))
        assert(isinstance(bert_opinions, BertOpinionsFormatter))

        for linked_df in self._iter_linked_opinions_df(news_id=news_id):
            assert(isinstance(linked_df, pd.DataFrame))
            # TODO. group also by binary labels in order to determite a result label
            pass

    def __group_by_binary_labels(self):
        # TODO.
        pass


