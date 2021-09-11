from os.path import join

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.input.readers.tsv_opinion import TsvInputOpinionReader
from arekit.common.experiment.input.readers.tsv_sample import TsvInputSampleReader
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

    def create_samples_reader(self, data_type):
        samples_tsv_filepath = self.get_input_sample_filepath(data_type)
        return TsvInputSampleReader.from_tsv(filepath=samples_tsv_filepath,
                                             row_ids_provider=MultipleIDProvider())

    def create_opinions_reader(self, data_type):
        opinions_tsv_filepath = self.get_input_opinions_filepath(data_type)
        return TsvInputOpinionReader.from_tsv(opinions_tsv_filepath, compression='infer')

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        """ Utilized for results evaluation.
        """
        assert(isinstance(epoch_index, int))

        model_eval_root = self.__get_eval_root_filepath(data_type=data_type, epoch_index=epoch_index)

        filepath = join(model_eval_root, "{}.opin.txt".format(doc_id))

        return filepath

    # region private methods

    def __get_eval_root_filepath(self, data_type, epoch_index):
        assert(isinstance(data_type, DataType))
        assert(isinstance(epoch_index, int))

        result_dir = join(
            self.get_target_dir(),
            join("eval/{data_type}/{iter_index}/{epoch_index}".format(
                data_type=data_type.name,
                iter_index=self._experiment_iter_index(),
                epoch_index=str(epoch_index))))

        return result_dir

    # endregion

    # TODO. In nested class (user applications)
    def get_input_opinions_filepath(self, data_type):
        template = self._filename_template(data_type=data_type)
        return self._get_filepath(out_dir=self.get_target_dir(),
                                  template=template,
                                  # TODO. formatter_type_log_name -- in nested formatter.
                                  prefix="opinion")

    # TODO. In nested class (user applications)
    def get_input_sample_filepath(self, data_type):
        template = self._filename_template(data_type=data_type)
        return self._get_filepath(out_dir=self.get_target_dir(),
                                  template=template,
                                  # TODO. formatter_type_log_name -- in nested formatter.
                                  prefix="sample")

    # TODO. In nested class (user applications)
    @staticmethod
    def _get_filepath(out_dir, template, prefix):
        assert(isinstance(template, str))
        assert(isinstance(prefix, str))
        return join(out_dir, BertIOUtils.__generate_tsv_archive_filename(template=template, prefix=prefix))

    # TODO. In nested class (user applications)
    def _experiment_iter_index(self):
        return self._experiment.DocumentOperations.DataFolding.IterationIndex

    # TODO. In nested class (user applications)
    def _filename_template(self, data_type):
        assert(isinstance(data_type, DataType))
        return "{data_type}-{iter_index}".format(data_type=data_type.name.lower(),
                                                 iter_index=self._experiment_iter_index())

    # TODO. In nested class (user applications)
    @staticmethod
    def __generate_tsv_archive_filename(template, prefix):
        return "{prefix}-{template}.tsv.gz".format(prefix=prefix, template=template)
