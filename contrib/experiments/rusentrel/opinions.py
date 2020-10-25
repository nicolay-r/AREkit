import collections
import logging
import os

from arekit.common.evaluation.utils import OpinionCollectionsToCompareUtils
from arekit.common.experiment.data.base import DataIO
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.cv_based.opinions import CVBasedOpinionOperations
from arekit.contrib.experiments.rusentrel.labels_formatter import RuSentRelNeutralLabelsFormatter
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinionCollection


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RuSentrelOpinionOperations(CVBasedOpinionOperations):

    def __init__(self, data_io, version, neutral_root, rusentrel_news_ids):
        assert(isinstance(data_io, DataIO))
        assert(isinstance(version, RuSentRelVersions))
        assert(isinstance(rusentrel_news_ids, set))

        super(RuSentrelOpinionOperations, self).__init__(model_io=data_io.ModelIO,
                                                         folding_algo=data_io.CVFoldingAlgorithm)

        self._set_synonyms_collection(data_io.SynonymsCollection)
        self._set_neutral_root(neutral_root)

        self.__opinion_formatter = data_io.OpinionFormatter
        self._rusentrel_news_ids = rusentrel_news_ids
        self.__eval_on_rusentrel_docs_key = True
        self.__result_labels_fmt = RuSentRelLabelsFormatter()
        self.__neutral_labels_fmt = RuSentRelNeutralLabelsFormatter()
        self._rusentrel_version = version

    # region property

    @property
    def NewsIDs(self):
        return self._rusentrel_news_ids

    # endregion

    # region CVBasedOperations

    def get_doc_ids_set_to_neutrally_annotate(self):
        # Note:
        # We provide neutral annotation for every
        # document of RuSentRelCollection.
        return self._rusentrel_news_ids

    def get_doc_ids_set_to_compare(self):
        return self._rusentrel_news_ids

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))
        return RuSentRelOpinionCollection.load_collection(doc_id=doc_id,
                                                          synonyms=self._synonyms,
                                                          version=self._rusentrel_version)

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        """
        Note: Assumes that all the results already converted into document-level opinions.
        """
        assert(isinstance(data_type, DataType))
        assert(isinstance(doc_ids, collections.Iterable))
        assert(isinstance(epoch_index, int))

        opinions_cmp_iter = OpinionCollectionsToCompareUtils.iter_comparable_collections(
            doc_ids=filter(lambda doc_id: doc_id in self.get_doc_ids_set_to_compare(), doc_ids),
            read_etalon_collection_func=lambda doc_id: self.read_etalon_opinion_collection(doc_id=doc_id),
            read_result_collection_func=lambda doc_id: self.__opinion_formatter.load_from_file(
                filepath=self.create_result_opinion_collection_filepath(data_type=data_type,
                                                                        doc_id=doc_id,
                                                                        epoch_index=epoch_index),
                labels_formatter=self.__result_labels_fmt))

        for opinions_cmp in opinions_cmp_iter:
            yield opinions_cmp

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(data_type, DataType))

        filepath = self.create_neutral_opinion_collection_filepath(doc_id=doc_id,
                                                                   data_type=data_type)

        if not os.path.exists(filepath):
            logger.info("Neutral collection does not exists: {}".format(filepath))
            logger.info("Providing empty one instead")
            return self.create_opinion_collection()

        return self.__opinion_formatter.load_from_file(filepath=filepath,
                                                       labels_formatter=self.__neutral_labels_fmt)

    # endregion