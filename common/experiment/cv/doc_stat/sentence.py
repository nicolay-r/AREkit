from arekit.common.experiment.cv.doc_stat.base import BaseDocumentStatGenerator
from arekit.common.news.base import News


class SentenceBasedDocumentStatGenerator(BaseDocumentStatGenerator):

    def __calc(self, news):
        assert(isinstance(news, News))
        return news.SentencesCount
