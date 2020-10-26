from arekit.common.experiment.cv.doc_stat.base import DocStatGeneratorBase
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions
from arekit.contrib.source.rusentrel.news.base import RuSentRelNews


class RuSentRelDocStatGenerator(DocStatGeneratorBase):

    def __init__(self, synonyms, version):
        assert(isinstance(version, RuSentRelVersions))
        self.__synonyms = synonyms
        self.__version = version

    def _iter_doc_ids(self):
        for doc_id in RuSentRelIOUtils.iter_collection_indices():
            yield doc_id

    # TODO. Provide reader here, in order to deal with news
    # TODO. rather than utilize a specific document reader.
    def _calculate_sentences_count(self, doc_id):

        news = RuSentRelNews.read_document(doc_id=doc_id,
                                           synonyms=self.__synonyms,
                                           version=self.__version)

        return news.SentencesCount
