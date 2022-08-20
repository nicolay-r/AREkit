from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.synonyms.base import SynonymsCollection
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.news_reader import RuSentRelNewsReader


class RuSentrelDocumentOperations(DocumentOperations):
    """ Limitations: Supported only train/test collections format
    """

    def __init__(self, version, synonyms):
        assert(isinstance(version, RuSentRelVersions))
        assert(isinstance(synonyms, SynonymsCollection))
        super(RuSentrelDocumentOperations, self).__init__()
        self.__version = version
        self.__synonyms = synonyms

    def get_doc(self, doc_id):
        assert (isinstance(doc_id, int))
        return RuSentRelNewsReader.read_document(doc_id=doc_id, synonyms=self.__synonyms, version=self.__version)

