from arekit.common.experiment.api.ctx_base import DataIO
from arekit.common.experiment.api.ctx_serialization import SerializationData
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.news.base import RuSentRelNews
from arekit.contrib.source.rusentrel.news.parse_options import RuSentRelNewsParseOptions


class RuSentrelDocumentOperations(DocumentOperations):
    """
    Limitations: Supported only train/test collections format
    """

    def __init__(self, exp_data, folding, version, get_synonyms_func):
        assert(isinstance(exp_data, DataIO))
        assert(isinstance(version, RuSentRelVersions))
        assert(callable(get_synonyms_func))
        super(RuSentrelDocumentOperations, self).__init__(folding=folding)
        # TODO. exp_data should be removed.
        self.__exp_data = exp_data
        self.__version = version
        self.__get_synonyms_func = get_synonyms_func

    # region DocumentOperations

    def iter_doc_ids_to_annotate(self):
        return self.DataFolding.iter_doc_ids()

    def iter_doc_ids_to_compare(self):
        return self.DataFolding.iter_doc_ids()

    def read_news(self, doc_id):
        assert(isinstance(doc_id, int))
        synonyms = self.__get_synonyms_func()
        return RuSentRelNews.read_document(doc_id=doc_id,
                                           synonyms=synonyms,
                                           version=self.__version)

    # TODO. This should be removed, since parse-options considered as a part
    # TODO. Of the text-parser instance!!!
    # TODO. Parse options should not be related to the particular collection.
    def _create_parse_options(self):
        assert(isinstance(self.__exp_data, SerializationData))
        return RuSentRelNewsParseOptions(stemmer=self.__exp_data.Stemmer,
                                         frame_variants_collection=self.__exp_data.FrameVariantCollection)

    # endregion