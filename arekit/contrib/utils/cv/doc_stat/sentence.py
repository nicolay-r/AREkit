from arekit.common.news.base import News
from arekit.contrib.utils.cv.doc_stat.base import BaseDocumentStatGenerator


class SentenceBasedDocumentStatGenerator(BaseDocumentStatGenerator):

    def __init__(self, doc_reader_func):
        super(SentenceBasedDocumentStatGenerator, self).__init__(doc_reader_func)

    def _calc(self, news):
        assert(isinstance(news, News))
        return news.SentencesCount
