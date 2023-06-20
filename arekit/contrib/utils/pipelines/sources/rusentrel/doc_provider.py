from arekit.common.data.doc_provider import DocumentProvider
from arekit.common.synonyms.base import SynonymsCollection
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.docs_reader import RuSentRelDocumentsReader


class RuSentrelDocumentProvider(DocumentProvider):
    """ Limitations: Supported only train/test collections format
    """

    def __init__(self, version, synonyms):
        assert(isinstance(version, RuSentRelVersions))
        assert(isinstance(synonyms, SynonymsCollection))
        super(RuSentrelDocumentProvider, self).__init__()
        self.__version = version
        self.__synonyms = synonyms

    def by_id(self, doc_id):
        assert (isinstance(doc_id, int))
        return RuSentRelDocumentsReader.read_document(doc_id=doc_id, synonyms=self.__synonyms, version=self.__version)

