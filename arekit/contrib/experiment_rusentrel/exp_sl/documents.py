from arekit.common.experiment.api.enums import BaseDocumentTag
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.news_reader import RuSentRelNews


class RuSentrelDocumentOperations(DocumentOperations):
    """
    Limitations: Supported only train/test collections format
    """

    def __init__(self, exp_ctx, text_parser, version, get_synonyms_func):
        assert(isinstance(version, RuSentRelVersions))
        assert(callable(get_synonyms_func))
        super(RuSentrelDocumentOperations, self).__init__(exp_ctx=exp_ctx, text_parser=text_parser)

        self.__version = version
        self.__get_synonyms_func = get_synonyms_func

    # region DocumentOperations

    def iter_tagget_doc_ids(self, tag):
        assert(isinstance(tag, BaseDocumentTag))
        assert(tag == BaseDocumentTag.Compare or tag == BaseDocumentTag.Annotate)
        return self.DataFolding.iter_doc_ids()

    def get_doc(self, doc_id):
        assert(isinstance(doc_id, int))
        synonyms = self.__get_synonyms_func()
        return RuSentRelNews.read_document(doc_id=doc_id,
                                           synonyms=synonyms,
                                           version=self.__version)

    # endregion