import itertools

from arekit.common.experiment.data_io import DataIO
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.cv_based.documents import CVBasedDocumentOperations
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.source.rusentrel.news.base import RuSentRelNews
from arekit.source.rusentrel.news.parse_options import RuSentRelNewsParseOptions


class RuSentrelDocumentOperations(CVBasedDocumentOperations):
    """
    Limitations: Supported only train/test collections format
    """

    def __init__(self, data_io):
        assert(isinstance(data_io, DataIO))
        super(RuSentrelDocumentOperations, self).__init__(folding_algo=data_io.CVFoldingAlgorithm)
        self.__data_io = data_io

    def __use_fixed_folding(self):
        return self._FoldingAlgo.CVCount == 1

    def get_fixed_folding(self, data_type):
        if data_type == DataType.Train:
            return RuSentRelIOUtils.iter_train_indices()
        elif data_type == DataType.Test:
            return RuSentRelIOUtils.iter_test_indices()
        else:
            raise NotImplementedError("DataType '{}' is not supported".format(data_type))

    def get_data_indices_to_fold(self):
        return list(itertools.chain(RuSentRelIOUtils.iter_train_indices(),
                                    RuSentRelIOUtils.iter_test_indices()))

    def iter_news_indices(self, data_type):
        if self.__use_fixed_folding():
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
                                           synonyms=self.__data_io.SynonymsCollection)

    def create_parse_options(self):
        return RuSentRelNewsParseOptions(keep_tokens=self.__data_io.KeepTokens,
                                         stemmer=self.__data_io.Stemmer)
