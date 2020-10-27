from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.synonyms import SynonymsCollection
from arekit.contrib.experiments.ruattitudes.opinions import RuAttitudesOpinionOperations
from arekit.contrib.experiments.rusentrel.opinions import RuSentrelOpinionOperations


class RuSentrelWithRuAttitudesOpinionOperations(OpinionOperations):

    def __init__(self, synonyms, rusentrel_op, ruattitudes_op):
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(rusentrel_op, RuSentrelOpinionOperations))
        assert(isinstance(ruattitudes_op, RuAttitudesOpinionOperations))

        super(RuSentrelWithRuAttitudesOpinionOperations, self).__init__()

        self.__rusentrel_op = rusentrel_op
        self.__ruattitudes_op = ruattitudes_op

    # region CVBasedOpinionOperations

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))
        if doc_id in self.__rusentrel_op.NewsIDs:
            return self.__rusentrel_op.read_etalon_opinion_collection(doc_id)
        else:
            return self.__ruattitudes_op.read_etalon_opinion_collection(doc_id)

    def try_read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))
        if doc_id not in self.__rusentrel_op.NewsIDs and data_type == DataType.Train:
            return self.__ruattitudes_op.try_read_neutral_opinion_collection(doc_id=doc_id,
                                                                             data_type=data_type)
        else:
            return self.__rusentrel_op.try_read_neutral_opinion_collection(doc_id=doc_id,
                                                                           data_type=data_type)

    def get_doc_ids_set_to_compare(self):
        return self.__rusentrel_op.get_doc_ids_set_to_compare()

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        return self.__rusentrel_op.iter_opinion_collections_to_compare(data_type=data_type,
                                                                       doc_ids=doc_ids,
                                                                       epoch_index=epoch_index)

    def get_doc_ids_set_to_neutrally_annotate(self):
        return self.__rusentrel_op.get_doc_ids_set_to_neutrally_annotate()

    # endregion