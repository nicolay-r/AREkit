import os

from arekit.common.experiment.cv.base import BaseCVFolding
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.model.model_io import BaseModelIO
from arekit.common.utils import create_dir_if_not_exists


class CVBasedOpinionOperations(OpinionOperations):

    def __init__(self, model_io, folding_algo, neutral_root):
        assert(isinstance(model_io, BaseModelIO))
        assert(isinstance(folding_algo, BaseCVFolding))

        super(CVBasedOpinionOperations, self).__init__(neutral_root=neutral_root)

        self.__model_io = model_io
        self.__folding_algo = folding_algo

    # region private methods

    def __get_eval_root_filepath(self, data_type, epoch_index):
        assert(isinstance(data_type, DataType))
        assert(isinstance(epoch_index, int))

        result_dir = os.path.join(
            self.__model_io.ModelRoot,
            os.path.join(u"eval/{data_type}/{iter_index}/{epoch_index}".format(
                data_type=data_type.name,
                iter_index=self.__folding_algo.IterationIndex,
                epoch_index=str(epoch_index))))

        create_dir_if_not_exists(result_dir)
        return result_dir

    # endregion

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        assert(isinstance(epoch_index, int))

        model_eval_root = self.__get_eval_root_filepath(data_type=data_type,
                                                        epoch_index=epoch_index)

        filepath = os.path.join(model_eval_root, u"{}.opin.txt".format(doc_id))
        create_dir_if_not_exists(filepath)
        return filepath