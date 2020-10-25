import os

from arekit.common.experiment.cv.base import BaseCVFolding
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.model.model_io import BaseModelIO


class CVBasedOpinionOperations(OpinionOperations):

    def __init__(self, get_model_io_func, folding_algo):
        assert(callable(get_model_io_func))
        assert(isinstance(folding_algo, BaseCVFolding))

        super(CVBasedOpinionOperations, self).__init__()

        self.__get_model_io_func = get_model_io_func
        self.__folding_algo = folding_algo

    # region private methods

    def __get_eval_root_filepath(self, data_type, epoch_index):
        assert(isinstance(data_type, DataType))
        assert(isinstance(epoch_index, int))

        model_io = self.__get_model_io_func()
        assert(isinstance(model_io, BaseModelIO))

        result_dir = os.path.join(
            model_io.get_model_dir(),
            os.path.join(u"eval/{data_type}/{iter_index}/{epoch_index}".format(
                data_type=data_type.name,
                iter_index=self.__folding_algo.IterationIndex,
                epoch_index=str(epoch_index))))

        return result_dir

    # endregion

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        assert(isinstance(epoch_index, int))

        model_eval_root = self.__get_eval_root_filepath(data_type=data_type,
                                                        epoch_index=epoch_index)

        filepath = os.path.join(model_eval_root, u"{}.opin.txt".format(doc_id))

        return filepath
