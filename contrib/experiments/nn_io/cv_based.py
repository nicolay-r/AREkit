import os

from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.experiments.cv.utils import get_cv_pair_by_index
from arekit.contrib.experiments.experiment_io import BaseExperimentNeuralNetworkIO


class CVBasedNeuralNetworkIO(BaseExperimentNeuralNetworkIO):

    def __init__(self, model_name, cv_count, data_io):
        assert(isinstance(cv_count, int))
        super(CVBasedNeuralNetworkIO, self).__init__(data_io=data_io,
                                                     model_name=model_name)
        self.__current_cv_index = 0
        self.__cv_count = cv_count
        self.__docs_stat = self.create_docs_stat_generator()

    # region properties

    @property
    def CVCurrentIndex(self):
        return self.__current_cv_index

    @property
    def CVCount(self):
        return self.__cv_count

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

    def create_docs_stat_generator(self):
        raise NotImplementedError()

    def inc_cv_index(self):
        self.__current_cv_index += 1

    def iter_train_data_indices(self):
        train, _ = get_cv_pair_by_index(cv_count=self.__cv_count,
                                        cv_index=self.__current_cv_index,
                                        data_io=self.DataIO,
                                        docs_stat=self.__docs_stat)
        for doc_id in train:
            yield doc_id

    def iter_test_data_indices(self):
        _, test = get_cv_pair_by_index(cv_count=self.__cv_count,
                                       cv_index=self.__current_cv_index,
                                       data_io=self.DataIO,
                                       docs_stat=self.__docs_stat)
        for doc_id in test:
            yield doc_id

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        assert(isinstance(epoch_index, int))

        model_eval_root = self.__get_eval_root_filepath(data_type=data_type,
                                                        epoch_index=epoch_index)

        filepath = os.path.join(model_eval_root, u"{}.opin.txt".format(doc_id))
        create_dir_if_not_exists(filepath)
        return filepath

