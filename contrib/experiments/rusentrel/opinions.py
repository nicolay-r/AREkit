import collections
import os

from arekit.common.experiment.data_io import DataIO
from arekit.common.evaluation.utils import OpinionCollectionsToCompareUtils
from arekit.common.experiment.formats.cv_based.opinions import CVBasedOpinionOperations
from arekit.common.opinions.collection import OpinionCollection
from arekit.source.rusentrel.opinions.collection import RuSentRelOpinionCollection


class RuSentrelOpinionOperations(CVBasedOpinionOperations):

    def __init__(self, data_io, annot_name_func, rusentrel_news_ids):
        assert(isinstance(data_io, DataIO))
        assert(isinstance(rusentrel_news_ids, set))

        super(RuSentrelOpinionOperations, self).__init__(
            model_root=data_io.get_model_root(),
            experiments_dir=data_io.get_experiments_dir(),
            folding_algo=data_io.CVFoldingAlgorithm,
            annot_name_func=annot_name_func
        )

        self._data_io = data_io
        self._rusentrel_news_ids = rusentrel_news_ids
        self.__eval_on_rusentrel_docs_key = True

    def __iter_doc_ids_to_compare(self, doc_ids):
        if self.__eval_on_rusentrel_docs_key:
            return [doc_id for doc_id in doc_ids if doc_id in self._rusentrel_news_ids]

        return doc_ids

    def iter_doc_ids_to_compare(self, doc_ids):
        return self.__iter_doc_ids_to_compare(doc_ids)

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))
        return RuSentRelOpinionCollection.load_collection(doc_id=doc_id,
                                                          synonyms=self._data_io.SynonymsCollection)

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        assert(isinstance(data_type, unicode))
        assert(isinstance(doc_ids, collections.Iterable))
        assert(isinstance(epoch_index, int))

        opinions_cmp_iter = OpinionCollectionsToCompareUtils.iter_comparable_collections(
            doc_ids=self.__iter_doc_ids_to_compare(doc_ids),
            read_etalon_collection_func=lambda doc_id: self.read_etalon_opinion_collection(doc_id=doc_id),
            read_result_collection_func=lambda doc_id: self._data_io.OpinionFormatter.load_from_file(
                filepath=self.create_result_opinion_collection_filepath(data_type=data_type,
                                                                        doc_id=doc_id,
                                                                        epoch_index=epoch_index),
                synonyms=self._data_io.SynonymsCollection))

        for opinions_cmp in opinions_cmp_iter:
            yield opinions_cmp

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(data_type, unicode))

        filepath = self.create_neutral_opinion_collection_filepath(
            doc_id=doc_id,
            data_type=data_type)

        if not os.path.exists(filepath):
            return None

        return self._data_io.OpinionFormatter.load_from_file(filepath=filepath,
                                                             synonyms=self._data_io.SynonymsCollection)

    def create_opinion_collection(self, opinions=None):
        assert(isinstance(opinions, list) or opinions is None)
        return OpinionCollection(opinions=[] if opinions is None else opinions,
                                 synonyms=self._data_io.SynonymsCollection)

