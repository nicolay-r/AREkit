from arekit.common.experiment.cv.doc_stat.base import BaseDocumentStatGenerator
from arekit.common.news.base import News


# TODO. depends on io, issue #189
# TODO. We should not adopt inheritance there!
# TODO. Move into the particular experiment issue #189
class SentenceBasedDocumentStatGenerator(BaseDocumentStatGenerator):

    def __init__(self, doc_reader_func):
        super(SentenceBasedDocumentStatGenerator, self).__init__(doc_reader_func)

    def _calc(self, news):
        assert(isinstance(news, News))
        return news.SentencesCount
