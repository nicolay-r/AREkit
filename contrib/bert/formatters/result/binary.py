from arekit.contrib.bert.formatters.result.base import BertResults


class BertBinaryResults(BertResults):

    def __init__(self, df):
        super(BertBinaryResults, self).__init__(df=df)
        df.column = [u'yes', u'no']

