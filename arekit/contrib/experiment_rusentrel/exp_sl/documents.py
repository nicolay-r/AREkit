from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.news_reader import RuSentRelNewsReader


class RuSentrelDocumentOperations(DocumentOperations):
    """
    Limitations: Supported only train/test collections format
    """

    def __init__(self, version, get_synonyms_func):
        assert(isinstance(version, RuSentRelVersions))
        assert(callable(get_synonyms_func))
        super(RuSentrelDocumentOperations, self).__init__()

        self.__version = version
        self.__get_synonyms_func = get_synonyms_func

    def get_doc(self, doc_id):
        assert(isinstance(doc_id, int))
        synonyms = self.__get_synonyms_func()
        return RuSentRelNewsReader.read_document(doc_id=doc_id,
                                                 synonyms=synonyms,
                                                 version=self.__version)
