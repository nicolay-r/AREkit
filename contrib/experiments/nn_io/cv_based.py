import os

from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.experiments.experiment_io import BaseExperimentNeuralNetworkIO
from arekit.networks.data_type import DataType


class CVBasedNeuralNetworkIO(BaseExperimentNeuralNetworkIO):

    def __init__(self, model_name, data_io):
        super(CVBasedNeuralNetworkIO, self).__init__(data_io=data_io,
                                                     model_name=model_name)

    # region private methods

    def __get_eval_root_filepath(self, data_type, epoch_index):
        assert(isinstance(epoch_index, int))

        result_dir = os.path.join(
            self.get_model_root(),
            os.path.join(u"eval/{data_type}/{iter_index}/{epoch_index}".format(
                data_type=data_type,
                iter_index=self.DataIO.CVFoldingAlgorithm.IterationIndex,
                epoch_index=str(epoch_index))))

        create_dir_if_not_exists(result_dir)
        return result_dir

    # endregion

    def iter_data_indices(self, data_type):
        train, test = self.DataIO.CVFoldingAlgorithm.get_cv_pair_by_index()

        if data_type not in [DataType.Train, DataType.Test]:
            raise Exception("Not supported data_type='{data_type}'".format(data_type=data_type))

        result_list = train if data_type == DataType.Train else test

        for doc_id in result_list:
            yield doc_id

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        assert(isinstance(epoch_index, int))

        model_eval_root = self.__get_eval_root_filepath(data_type=data_type,
                                                        epoch_index=epoch_index)

        filepath = os.path.join(model_eval_root, u"{}.opin.txt".format(doc_id))
        create_dir_if_not_exists(filepath)
        return filepath

