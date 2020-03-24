import os

from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.experiments.experiment_io import BaseExperimentNeuralNetworkIO


class CVBasedNeuralNetworkIO(BaseExperimentNeuralNetworkIO):

    def __init__(self, model_name, data_io):
        super(CVBasedNeuralNetworkIO, self).__init__(data_io=data_io,
                                                     model_name=model_name)
        self.__current_cv_index = 0

    # region properties

    @property
    def CVCurrentIndex(self):
        return self.__current_cv_index

    # endregion

    # region private methods

    def __get_eval_root_filepath(self, data_type, epoch_index):
        assert(isinstance(epoch_index, int))

        result_dir = os.path.join(
            self.get_model_root(),
            os.path.join(u"eval/{}/{}/{}".format(data_type,
                                                 self.__current_cv_index,
                                                 str(epoch_index))))

        create_dir_if_not_exists(result_dir)
        return result_dir

    # endregion

    # TODO. Refactor
    # TODO. Leave a single method
    def iter_train_data_indices(self):
        train, _ = self.DataIO.CVFoldingAlgorithm.get_cv_pair_by_index()

        for doc_id in train:
            yield doc_id

    # TODO. Refactor
    # TODO. Leave a single method
    def iter_test_data_indices(self):
        _, test = self.DataIO.CVFoldingAlgorithm.get_cv_pair_by_index()

        for doc_id in test:
            yield doc_id

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        assert(isinstance(epoch_index, int))

        model_eval_root = self.__get_eval_root_filepath(data_type=data_type,
                                                        epoch_index=epoch_index)

        filepath = os.path.join(model_eval_root, u"{}.opin.txt".format(doc_id))
        create_dir_if_not_exists(filepath)
        return filepath

