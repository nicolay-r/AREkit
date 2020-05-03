import collections
import itertools
import os

from arekit.common.experiment.data_type import DataType
from arekit.common.evaluation.utils import OpinionCollectionsToCompareUtils
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.experiment.cv_based import CVBasedExperiment
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.source.rusentrel.news.base import RuSentRelNews
from arekit.source.rusentrel.news.parse_options import RuSentRelNewsParseOptions
from arekit.source.rusentrel.opinions.collection import RuSentRelOpinionCollection


# TODO. Rename as experiment
class RuSentRelExperiment(CVBasedExperiment):
    """
    Represents Input interface for NeuralNetwork ctx
    Now exploited (treated) as an input interface only
    """

    def __init__(self, data_io, prepare_model_root):
        super(RuSentRelExperiment, self).__init__(data_io=data_io,
                                                  prepare_model_root=prepare_model_root)

        self.__rusentrel_news_ids_list = list(RuSentRelIOUtils.iter_collection_indices())
        self.__rusentrel_news_ids = set(self.__rusentrel_news_ids_list)

        self.__eval_on_rusentrel_docs_key = True

    # region properties

    @property
    def RuSentRelNewsIDsList(self):
        return self.__rusentrel_news_ids_list

    # endregion

    def is_rusentrel_news_id(self, news_id):
        assert(isinstance(news_id, int))
        return news_id in self.__rusentrel_news_ids

    # region 'read' public methods

    def read_news(self, doc_id):
        assert(isinstance(doc_id, int))
        return RuSentRelNews.read_document(doc_id=doc_id, synonyms=self.DataIO.SynonymsCollection)

    def create_parse_options(self):
        return RuSentRelNewsParseOptions(keep_tokens=self.DataIO.KeepTokens,
                                         stemmer=self.DataIO.Stemmer)

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(data_type, unicode))

        filepath = self.create_neutral_opinion_collection_filepath(
            doc_id=doc_id,
            data_type=data_type)

        if not os.path.exists(filepath):
            return None

        return self.DataIO.OpinionFormatter.load_from_file(filepath=filepath,
                                                           synonyms=self.DataIO.SynonymsCollection)

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))
        return RuSentRelOpinionCollection.load_collection(doc_id=doc_id,
                                                          synonyms=self.DataIO.SynonymsCollection)

    # endregion

    # region 'iters' public methods

    def __use_fixed_folding(self):
        return self.DataIO.CVFoldingAlgorithm.CVCount == 1

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
            for doc_id in super(RuSentRelExperiment, self).iter_news_indices(data_type):
                yield doc_id

    def __iter_doc_ids_to_compare(self, doc_ids):
        if self.__eval_on_rusentrel_docs_key:
            return [doc_id for doc_id in doc_ids if doc_id in self.__rusentrel_news_ids]

        return doc_ids

    def iter_doc_ids_to_compare(self, doc_ids):
        return self.__iter_doc_ids_to_compare(doc_ids)

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        assert(isinstance(data_type, unicode))
        assert(isinstance(doc_ids, collections.Iterable))
        assert(isinstance(epoch_index, int))

        opinions_cmp_iter = OpinionCollectionsToCompareUtils.iter_comparable_collections(
            doc_ids=self.__iter_doc_ids_to_compare(doc_ids),
            read_etalon_collection_func=lambda doc_id: self.read_etalon_opinion_collection(doc_id=doc_id),
            read_result_collection_func=lambda doc_id: self.DataIO.OpinionFormatter.load_from_file(
                filepath=self.create_result_opinion_collection_filepath(data_type=data_type,
                                                                        doc_id=doc_id,
                                                                        epoch_index=epoch_index),
                synonyms=self.DataIO.SynonymsCollection))

        for opinions_cmp in opinions_cmp_iter:
            yield opinions_cmp

    # endregion

    # region 'create' public methods

    def create_opinion_collection(self, opinions=None):
        assert(isinstance(opinions, list) or opinions is None)
        return OpinionCollection(opinions=[] if opinions is None else opinions,
                                 synonyms=self.DataIO.SynonymsCollection)

    # endregion
