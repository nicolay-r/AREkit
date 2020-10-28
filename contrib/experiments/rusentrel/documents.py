from arekit.common.experiment.data.base import DataIO
from arekit.common.experiment.data.serializing import SerializationData
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.documents_cv_based import CVBasedDocumentOperations
from arekit.contrib.experiments.rusentrel.folding_type import FoldingType
from arekit.contrib.experiments.rusentrel.utils import get_rusentrel_inds
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.news.base import RuSentRelNews
from arekit.contrib.source.rusentrel.news.parse_options import RuSentRelNewsParseOptions


class RuSentrelDocumentOperations(CVBasedDocumentOperations):
    """
    Limitations: Supported only train/test collections format
    """

    def __init__(self, data_io, folding_type, version):
        assert(isinstance(data_io, DataIO))
        assert(isinstance(version, RuSentRelVersions))
        assert(isinstance(folding_type, FoldingType))
        super(RuSentrelDocumentOperations, self).__init__(folding_algo=data_io.CVFoldingAlgorithm)

        train, test, all = get_rusentrel_inds(version)

        self.__train_doc_ids = train
        self.__test_doc_ids = test
        self.__all_doc_ids = set(all)

        self.__data_io = data_io
        self.__version = version
        self.__doc_ids_to_neut_annot = self.__all_doc_ids
        self.__doc_ids_to_fold = self.__all_doc_ids
        self.__folding_type = folding_type

        self.__supported_types = [DataType.Train, DataType.Test]

    # region public methods

    def contains_doc_id(self, doc_id):
        return doc_id in self.__all_doc_ids

    def get_doc_ids(self):
        return self.__all_doc_ids

    # endregion

    # region DocumentOperations

    def get_fixed_folding(self, data_type):
        if data_type not in self.__supported_types:
            raise Exception("Not supported data_type='{data_type}'".format(data_type=data_type))

        if data_type == DataType.Train:
            return iter(self.__train_doc_ids)

        elif data_type == DataType.Test:
            return iter(self.__test_doc_ids)

    def get_doc_ids_set_to_neutrally_annotate(self):
        return self.__doc_ids_to_neut_annot

    def get_doc_ids_set_to_fold(self):
        return self.__doc_ids_to_fold

    def iter_supported_data_types(self):
        return iter(self.__supported_types)

    def iter_news_indices(self, data_type):
        if data_type not in self.__supported_types:
            raise Exception("Not supported data_type='{data_type}'".format(data_type=data_type))

        if self.__folding_type == FoldingType.Fixed:
            for doc_id in self.get_fixed_folding(data_type):
                yield doc_id

        elif self.__folding_type == FoldingType.CrossValidation:
            for doc_id in super(RuSentrelDocumentOperations, self).iter_news_indices(data_type):
                yield doc_id

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