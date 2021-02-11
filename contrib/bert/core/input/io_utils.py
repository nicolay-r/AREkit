from os.path import join

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.io_utils import BaseIOUtils


class BertIOUtils(BaseIOUtils):

    def get_target_dir(self):
        """ Provides a main directory for input

            NOTE:
            We consider to save serialized results into model dir,
            rather than experiment dir in a base implementation,
            as model affects on text_b, entities representation, etc.
        """
        return join(super(BertIOUtils, self).get_target_dir(),
                    self._experiment.DataIO.ModelIO.get_model_name())

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        """ Utilized for results evaluation.
        """
        assert(isinstance(epoch_index, int))

        model_eval_root = self.__get_eval_root_filepath(data_type=data_type, epoch_index=epoch_index)

        filepath = join(model_eval_root, u"{}.opin.txt".format(doc_id))

        return filepath

    # region private methods

    def __get_eval_root_filepath(self, data_type, epoch_index):
        assert(isinstance(data_type, DataType))
        assert(isinstance(epoch_index, int))

        result_dir = join(
            self.get_target_dir(),
            join(u"eval/{data_type}/{iter_index}/{epoch_index}".format(
                data_type=data_type.name,
                iter_index=self._experiment_iter_index(),
                epoch_index=str(epoch_index))))

        return result_dir

    # endregion