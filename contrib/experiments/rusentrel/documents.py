from arekit.common.experiment.data.base import DataIO
from arekit.common.experiment.data.serializing import SerializationData
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.contrib.experiments.rusentrel.utils import get_rusentrel_inds
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.news.base import RuSentRelNews
from arekit.contrib.source.rusentrel.news.parse_options import RuSentRelNewsParseOptions


class RuSentrelDocumentOperations(DocumentOperations):
    """
    Limitations: Supported only train/test collections format
    """

    def __init__(self, data_io, folding, version):
        assert(isinstance(data_io, DataIO))
        assert(isinstance(version, RuSentRelVersions))
        super(RuSentrelDocumentOperations, self).__init__(folding=folding)
        _, _, all = get_rusentrel_inds(version)
        self.__all_doc_ids = set(all)
        self.__data_io = data_io
        self.__version = version
        # TODO. To Annot.
        self.__doc_ids_to_neut_annot = self.__all_doc_ids

    # region DocumentOperations

    def get_doc_ids_set_to_neutrally_annotate(self):
        return self.__doc_ids_to_neut_annot

    def read_news(self, doc_id):
        assert(isinstance(doc_id, int))
        return RuSentRelNews.read_document(doc_id=doc_id,
                                           synonyms=self.__data_io.SynonymsCollection,
                                           version=self.__version)

    def _create_parse_options(self):
        assert(isinstance(self.__data_io, SerializationData))
        return RuSentRelNewsParseOptions(stemmer=self.__data_io.Stemmer,
                                         frame_variants_collection=self.__data_io.FrameVariantCollection)

    # endregion