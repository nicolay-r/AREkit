import os

from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.experiments.base import BaseExperiment
from arekit.common.data_type import DataType


class CVBasedExperiment(BaseExperiment):

    def __init__(self, data_io, prepare_model_root):
        super(CVBasedExperiment, self).__init__(data_io=data_io,
                                                prepare_model_root=prepare_model_root)

    # region private methods

    def __get_eval_root_filepath(self, data_type, epoch_index):
        assert(isinstance(epoch_index, int))

        result_dir = os.path.join(
            self.DataIO.get_model_root(),
            os.path.join(u"eval/{data_type}/{iter_index}/{epoch_index}".format(
                data_type=data_type,
                iter_index=self.DataIO.CVFoldingAlgorithm.IterationIndex,
                epoch_index=str(epoch_index))))

        create_dir_if_not_exists(result_dir)
        return result_dir

    # endregion

    def get_data_indices_to_fold(self):
        raise NotImplementedError()

    def iter_news_indices(self, data_type):
        data_indices = self.get_data_indices_to_fold()
        train, test = self.DataIO.CVFoldingAlgorithm.get_cv_train_test_pair_by_index(doc_ids_iter=data_indices)

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

