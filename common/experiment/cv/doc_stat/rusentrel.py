from arekit.common.experiment.cv.doc_stat.base import DocStatGeneratorBase
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.contrib.source.rusentrel.news.base import RuSentRelNews


class RuSentRelDocStatGenerator(DocStatGeneratorBase):

    def __init__(self, synonyms):
        self.__synonyms = synonyms

    def iter_doc_ids(self):
        for doc_id in RuSentRelIOUtils.iter_collection_indices():
            yield doc_id

    def calculate_sentences_count(self, doc_id):

        news = RuSentRelNews.read_document(doc_id=doc_id,
                                           synonyms=self.__synonyms)

        return news.SentencesCount
