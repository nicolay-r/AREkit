from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.experiment.utils import get_path_of_subfolder_in_experiments_dir
from arekit.contrib.experiments.ruattitudes.opinions import RuAttitudesOpinionOperations
from arekit.contrib.experiments.rusentrel.opinions import RuSentrelOpinionOperations


class RuSentrelWithRuAttitudesOpinionOperations(OpinionOperations):

    def __init__(self, data_io,  experiment_name, neutral_annot_name, rusentrel_op, ruattitudes_op):
        assert(isinstance(rusentrel_op, RuSentrelOpinionOperations))
        assert(isinstance(ruattitudes_op, RuAttitudesOpinionOperations))

        # TODO. Duplicated
        neutral_root = get_path_of_subfolder_in_experiments_dir(
            experiments_dir=data_io.get_input_samples_dir(experiment_name),
            subfolder_name=neutral_annot_name)

        super(RuSentrelWithRuAttitudesOpinionOperations, self).__init__(neutral_root=neutral_root)

        self.__rusentrel_op = rusentrel_op
        self.__ruattitudes_op = ruattitudes_op

    # region CVBasedOpinionOperations

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))
        if doc_id in self.__rusentrel_op.NewsIDs:
            return self.__rusentrel_op.read_etalon_opinion_collection(doc_id)
        else:
            return self.__ruattitudes_op.read_etalon_opinion_collection(doc_id)

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))
        if doc_id not in self.__rusentrel_op.NewsIDs and data_type == DataType.Train:
            return self.__ruattitudes_op.read_neutral_opinion_collection(doc_id=doc_id,
                                                                         data_type=data_type)
        else:
            return self.__rusentrel_op.read_neutral_opinion_collection(doc_id=doc_id,
                                                                       data_type=data_type)

    def get_doc_ids_set_to_compare(self, doc_ids):
        return self.__rusentrel_op.get_doc_ids_set_to_compare(doc_ids)

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        return self.__rusentrel_op.iter_opinion_collections_to_compare(data_type=data_type,
                                                                       doc_ids=doc_ids,
                                                                       epoch_index=epoch_index)

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        return self.__rusentrel_op.create_result_opinion_collection_filepath(data_type=data_type,
                                                                             doc_id=doc_id,
                                                                             epoch_index=epoch_index)

    # TODO. Weird
    def create_opinion_collection(self, opinions=None):
        return self.__rusentrel_op.create_opinion_collection()

    # endregion