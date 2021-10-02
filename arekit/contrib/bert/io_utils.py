from os.path import join, exists

from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.opinions import BaseOpinionStorageView
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.experiment.data_type import DataType
from arekit.common.utils import join_dir_with_subfolder_name


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

    def create_samples_view(self, data_type):
        return BaseSampleStorageView(
            storage=BaseRowsStorage.from_tsv(filepath=self.get_input_sample_filepath(data_type)),
            row_ids_provider=MultipleIDProvider())

    def create_opinions_view(self, data_type):
        storage = BaseRowsStorage.from_tsv(
            filepath=self.get_input_opinions_filepath(data_type),
            compression='infer')

        return BaseOpinionStorageView(storage=storage)

    def create_opinions_writer_target(self, data_type):
        return self.get_input_opinions_filepath(data_type)

    def create_samples_writer_target(self, data_type):
        return self.get_input_sample_filepath(data_type)

    def create_samples_writer(self):
        return TsvWriter(write_header=True)

    def create_opinions_writer(self):
        return TsvWriter(write_header=False)

    def create_result_opinion_collection_target(self, data_type, doc_id, epoch_index):
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

    # TODO. In nested class (user applications)
    def _get_annotator_name(self):
        """ We use custom implementation as it allows to
            be independent of NeutralAnnotator instance.
        """
        return "annot_{labels_count}l".format(labels_count=self._experiment.DataIO.LabelsCount)

    # TODO. In nested class (user applications)
    def __get_annotator_dir(self):
        return join_dir_with_subfolder_name(dir=self.get_target_dir(),
                                            subfolder_name=self._get_annotator_name())

    # TODO. In nested class (user applications)
    def _create_annotated_collection_target(self, doc_id, data_type, check_existance):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))
        assert(isinstance(check_existance, bool))

        annot_dir = self.__get_annotator_dir()

        if annot_dir is None:
            raise NotImplementedError("Neutral root was not provided!")

        # TODO. This should not depends on the neut.
        filename = "art{doc_id}.neut.{d_type}.txt".format(doc_id=doc_id,
                                                          d_type=data_type.name)

        target = join(annot_dir, filename)

        if check_existance and not exists(target):
            return None

        return target
