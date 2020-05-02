from arekit.contrib.bert.formatters.result.base import BertResults


class BertMultipleResults(BertResults):

    def __init__(self, df):
        super(BertMultipleResults, self).__init__(df=df)
        df.column = [u'neut', u'pos', u'neg']

