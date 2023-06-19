from arekit.common.docs.base import Document
from arekit.contrib.utils.cv.doc_stat.base import BaseDocumentStatGenerator


class SentenceBasedDocumentStatGenerator(BaseDocumentStatGenerator):

    def __init__(self, doc_reader_func):
        super(SentenceBasedDocumentStatGenerator, self).__init__(doc_reader_func)

    def _calc(self, news):
        assert(isinstance(news, Document))
        return news.SentencesCount
