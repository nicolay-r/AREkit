from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.contrib.experiments.ruattitudes.opinions import RuAttitudesOpinionOperations
from arekit.contrib.experiments.rusentrel.opinions import RuSentrelOpinionOperations


class RuSentrelWithRuAttitudesOpinionOperations(OpinionOperations):

    def __init__(self, synonyms, rusentrel_op, ruattitudes_op, is_rusentrel_doc):
        assert(isinstance(rusentrel_op, RuSentrelOpinionOperations))
        assert(isinstance(ruattitudes_op, RuAttitudesOpinionOperations))

        super(RuSentrelWithRuAttitudesOpinionOperations, self).__init__(synonyms)

        self.__rusentrel_op = rusentrel_op
        self.__ruattitudes_op = ruattitudes_op
        self.__is_rusentrel_doc = is_rusentrel_doc

    def __target(self, doc_id):
        if self.__is_rusentrel_doc(doc_id):
            return self.__rusentrel_op
        return self.__ruattitudes_op

    # region CVBasedOpinionOperations

    def iter_opinions_for_extraction(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        target = self.__target(doc_id)
        return target.iter_opinions_for_extraction(doc_id=doc_id,
                                                   data_type=data_type)

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))
        target = self.__target(doc_id)
        return target.read_etalon_opinion_collection(doc_id)

    def try_read_neutrally_annotated_opinion_collection(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))
        target = self.__target(doc_id)
        return target.try_read_neutrally_annotated_opinion_collection(doc_id=doc_id, data_type=data_type)

    def save_neutrally_annotated_opinion_collection(self, collection, doc_id, data_type):
        target = self.__target(doc_id)
        return target.save_neutrally_annotated_opinion_collection(collection=collection,
                                                                  doc_id=doc_id,
                                                                  data_type=data_type)

    def read_result_opinion_collection(self, data_type, doc_id, epoch_index):
        target = self.__target(doc_id)
        return target.read_result_opinion_collection(data_type=data_type,
                                                     doc_id=doc_id,
                                                     epoch_index=epoch_index)

    def create_opinion_collection(self):
        return self.__rusentrel_op.create_opinion_collection()

    # endregion