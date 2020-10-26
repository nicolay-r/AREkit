import itertools

from arekit.common.experiment.data.base import DataIO
from arekit.common.experiment.data.serializing import SerializationData
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.cv_based.documents import CVBasedDocumentOperations
from arekit.contrib.experiments.rusentrel.folding_type import FoldingType
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions
from arekit.contrib.source.rusentrel.news.base import RuSentRelNews
from arekit.contrib.source.rusentrel.news.parse_options import RuSentRelNewsParseOptions


class RuSentrelDocumentOperations(CVBasedDocumentOperations):
    """
    Limitations: Supported only train/test collections format
    """

    def __init__(self, data_io, folding_type, rusentrel_version):
        assert(isinstance(data_io, DataIO))
        assert(isinstance(rusentrel_version, RuSentRelVersions))
        assert(isinstance(folding_type, FoldingType))
        super(RuSentrelDocumentOperations, self).__init__(folding_algo=data_io.CVFoldingAlgorithm)
        self.__data_io = data_io
        self.__version = rusentrel_version
        self.__folding_type = folding_type

    def get_fixed_folding(self, data_type):
        if data_type == DataType.Train:
            return RuSentRelIOUtils.iter_train_indices()
        elif data_type == DataType.Test:
            return RuSentRelIOUtils.iter_test_indices()
        else:
            raise NotImplementedError("DataType '{}' is not supported".format(data_type))

    def iter_supported_data_types(self):
        yield DataType.Train
        yield DataType.Test

    def get_data_indices_to_fold(self):
        return list(itertools.chain(RuSentRelIOUtils.iter_train_indices(),
                                    RuSentRelIOUtils.iter_test_indices()))

    def iter_news_indices(self, data_type):
        if self.__folding_type == FoldingType.Fixed:
            if data_type not in [DataType.Train, DataType.Test]:
                raise Exception("Not supported data_type='{data_type}'".format(data_type=data_type))

            for doc_id in self.get_fixed_folding(data_type):
                yield doc_id
        else:
            for doc_id in super(RuSentrelDocumentOperations, self).iter_news_indices(data_type):
                yield doc_id

    def read_news(self, doc_id):
        assert(isinstance(doc_id, int))
        return RuSentRelNews.read_document(doc_id=doc_id,
                                           synonyms=self.__data_io.SynonymsCollection,
                                           version=self.__version)

    def create_parse_options(self):
        assert(isinstance(self.__data_io, SerializationData))
        return RuSentRelNewsParseOptions(stemmer=self.__data_io.Stemmer,
                                         frame_variants_collection=self.__data_io.FrameVariantCollection)
