from arekit.contrib.experiments.cv.docs_stat import DocStatGeneratorBase
from arekit.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.source.rusentrel.news import RuSentRelNews


class RuSentRelDocStatGenerator(DocStatGeneratorBase):

    def __init__(self, synonyms):
        self.__synonyms = synonyms

    def iter_doc_ids(self):
        for doc_id in RuSentRelIOUtils.iter_collection_indices():
            yield doc_id

    def calculate_sentences_count(self, doc_id):

        entities = RuSentRelDocumentEntityCollection.read_collection(
            doc_id=doc_id,
            synonyms=self.__synonyms)

        news = RuSentRelNews.read_document(doc_id=doc_id, entities=entities)

        yield (doc_id, news.SentencesCount())
